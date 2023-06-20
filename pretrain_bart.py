#config:utf-8

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer
import numpy as np
import json
import csv
from utils import to_device, Checkpoint, Step, Smoother, Logger
from config import build_config
from dataset import DAEdataset_DC
from transformers.optimization import get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup
from transformers import BertTokenizer
from EMA import ExponentialMovingAverage
from models import GenerationModel,GenerationModel_Pretrain
from torch.cuda.amp import autocast, GradScaler
import contextlib
from BlockShuffle import ChunkedBatchSampler

import os

@torch.no_grad()
def evaluate(model, loader, beam=1, n=-1):
    from rouge import Rouge
    rouge_score = Rouge()
    model.eval()
    all_pred = []
    all_targets = []
    for (source, targets, _) in tqdm(loader):
        source, targets = list(source), list(targets)
        with torch.no_grad():
            pred = model(source)
        pred = tokenizer.batch_decode(pred.cpu().numpy(), skip_special_tokens=True)
        print(pred)
        sys.exit(1)
        all_pred.extend(pred)
        all_targets.extend(targets)
    scores = rouge_score.get_scores(all_preds,all_targets,avg=True)
    for key in scores:
      scores[key] = scores[key]['f']*100
    result = scores
    print(result)
config = build_config()
tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name)

def pretrain(eval_epoch=3):
    config.output_dir = config.output_dir + '-pretrain'
    train_data = DAEdataset_DC(config.pretrain_file_list, config.input_l, config.output_l, tokenizer=tokenizer)
    valid_data = DAEdataset_DC(config.pretrain_file_list, config.input_l, config.output_l, tokenizer=tokenizer)

    train_chunked_batch_sampler = ChunkedBatchSampler(train_data, config.pretrain_batch, drop_last=False, shuffle=True)
    valid_chunked_batch_sampler = ChunkedBatchSampler(valid_data, config.valid_batch, drop_last=False, shuffle=False)
    train_loader = DataLoader(train_data, num_workers=1, batch_sampler=train_chunked_batch_sampler)
    valid_loader = DataLoader(valid_data, num_workers=1, batch_sampler=valid_chunked_batch_sampler)

    step = Step()
    model = GenerationModel_Pretrain(config)
    checkpoint = Checkpoint(model=model, step=step)
    if config.pretrained_checkpoint is not None:
        checkpoint.resume(config.pretrained_checkpoint)
    model = model.cuda()

    # num_training_steps = num_warmup_steps + (num_training_steps-num_warmup_steps)/(1-config['min_lr']/config['lr'])

    num_training_steps = config.pretrain_epoch * len(train_loader)
    num_warmup_steps = int(config.warmup_ratio * num_training_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.pretrain_lr)
    lr_schedule = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                  num_warmup_steps=num_warmup_steps,
                                                  num_training_steps=num_training_steps,
                                                  last_epoch=-1)
    start_epoch = 0
    start_epoch = config.pretrain_epoch - 1

    train_loss = Smoother(100)
    logger = Logger(config.output_dir + '/log.txt', 'a')
    logger.log(config)
    writer = SummaryWriter(config.output_dir)

    scaler = GradScaler()
    amp_cm = autocast if config.amp else contextlib.nullcontext
    Path(config.output_dir).mkdir(exist_ok=True, parents=True)
    if config.ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = config.finetune_batch * config.accumulation_steps * config.ema_steps / config.finetune_epoch
        alpha = 1.0 - config.ema_decay
        alpha = min(1.0, alpha * adjust)
        logger.log('EMA decay:', 1 - alpha)
        print('EMA decay:', 1 - alpha)
        model_ema = ExponentialMovingAverage(model, device='cuda', decay=1.0 - alpha)
        step_ema = Step()
        checkpoint_ema = Checkpoint(model=model_ema.module, step=step_ema)

    for epoch in range(start_epoch, config.pretrain_epoch):
        model.train()
        print('epoch:', epoch, 'lr:', optimizer.param_groups[0]['lr'])
        logger.log('new epoch', epoch)
        for iter, (source, target, target_noisy) in enumerate(tqdm(train_loader, dynamic_ncols=True)):

            source, target = list(source), list(target)
            target_noisy = list(target_noisy)
            step.forward(len(source))

            with amp_cm():
                pred, loss = model(source, target, target_noisy)
            if config.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            train_loss.update(loss={'celoss': loss.item()})

            if (iter + 1) % config.accumulation_steps_pretrain == 0:
                if config.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()  # 优化一次
                lr_schedule.step()
                optimizer.zero_grad()  # 清空梯度
                if config.ema and (iter + 1) % (config.accumulation_steps * config.ema_steps) == 0:
                    model_ema.update_parameters(model)
                    if epoch < int(config.ema_warmup_ratio * config.pretrain_epoch):
                        # Reset ema buffer to keep copying weights during warmup period
                        model_ema.n_averaged.fill_(0)
            if step.value % 100 == 0:
                logger.log(step.value, train_loss.value())

        if (epoch + 1) % eval_epoch == 0 or epoch == config.pretrain_epoch - 1:
            # checkpoint.save(config.output_dir+'/model_%d.pt'%epoch)
            metrics = evaluate(model, valid_loader)
        #     logger.log('valid', step.value, metrics.value())
        #     writer.add_scalars('valid metric', metrics.value(), step.value)
        #     checkpoint.update(config.output_dir + '/model.pt', metrics=metrics.value())
        #     if config.ema and epoch >= int(config.ema_warmup_ratio * config.pretrain_epoch):
        #         metrics_ema = evaluate(model_ema.module, valid_loader)
        #         logger.log('valid', step_ema.value, metrics_ema.value())
        #         writer.add_scalars('valid metric of ema', metrics_ema.value(), step_ema.value)
        #         checkpoint_ema.update(config.output_dir + '/model_ema.pt', metrics=metrics_ema.value())
        checkpoint.save(config.output_dir + '/model_last.pt')
        if config.ema and epoch >= int(config.ema_warmup_ratio * config.pretrain_epoch):
            checkpoint_ema.save(config.output_dir + '/model_ema_last.pt')
    logger.close()
    writer.close()

pretrain(1)
