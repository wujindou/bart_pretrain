#coding:utf-8
import json
import glob
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
from PIL import Image
import random
import time
import csv
import traceback

class BaseDataset(Dataset):
    def _try_getitem(self, idx):
        raise NotImplementedError
    def __getitem__(self, idx):
        wait = 0.1
        while True:
            try:
                ret = self._try_getitem(idx)
                return ret
            except KeyboardInterrupt:
                break
            except (Exception, BaseException) as e: 
                exstr = traceback.format_exc()
                print(exstr)
                print('read error, waiting:', wait)
                time.sleep(wait)
                wait = min(wait*2, 1000)
class DAEdataset(BaseDataset):
    def __init__(self, data_file_list, input_l, output_l,tokenizer):
        self.mask_token = tokenizer.mask_token
        self.description_token = tokenizer.description_token
        self.diagnosis_token = tokenizer.diagnosis_token
        self.clinical_token = tokenizer.clinical_token
        # self.clinical_token = tokenizer.sep_token        
        self.samples = []
        self.samples_prefix = []
        prefix = [self.description_token,self.clinical_token,self.diagnosis_token]
        for data_file in data_file_list:
            with open(data_file, 'r') as fp:
                reader = csv.reader(fp)
                for row in reader:
                    for k in range(1,len(row)):
                        if len(row[k].split())>0:
                            self.samples.append(row[k])
                            self.samples_prefix.append(prefix[k-1])
        self.tokenizer = tokenizer
        self.input_l = input_l
        self.output_l = output_l
    def __len__(self):
        return len(self.samples)
    def _try_getitem(self, idx):
        source = self.samples[idx]
        source_corrupted =self.data_corruption(source)
        source_noisy = self.token_replace(source,p=0.8)
        
        return source_corrupted,source,source_noisy       
    
    def text_infilling(self,my_list,span_ratio=0.15):
        #text infill
        #span_start 和 span_end 之间的text会被mask,会保留end
        temp_list = my_list.copy()
        arr_list = temp_list
        span_max_length = int(span_ratio*len(arr_list))
        span_total_length = 0
        arr_list_span = []
        span_end = 0
        while span_total_length < span_max_length and len(arr_list)>1:
            span_start = random.randint(0,(len(arr_list)-1)//2)
            span_length =  np.random.poisson(lam=3)
            span_length = min(span_length,span_max_length-span_total_length)
            span_end = min(span_start+span_length,len(arr_list)-1)
            if random.random()<0.1:
                arr_list_span = arr_list_span + arr_list[0:span_start]+[str(random.randint(9,1640))]
            else:
                arr_list_span = arr_list_span + arr_list[0:span_start]+[self.mask_token]
            arr_list = temp_list[span_end:]
            span_total_length += span_length

        arr_list_span += arr_list
        return arr_list_span
    
    def sentence_shuffle(self,my_list):
        arr_list = my_list.copy()
        sentence_list=[]
        sentence_end = []
        item=[]
        for i in range(len(arr_list)):
            item.append(arr_list[i])
            if arr_list[i] in sentence_end or i==len(arr_list)-1:
                sentence_list.append(item.copy())
                item = []
        random.shuffle(sentence_list)
        arr_list_shuffle = []
        for k in sentence_list:
            arr_list_shuffle.extend(k)
        return arr_list_shuffle

    def token_replace(self,my_str,p=1.0,p_drop=0.15):
        #text infill
        if random.random()>p:
            return my_str
        arr_list = [int(s) for s in my_str.split()]
                
        #drop as BERT
        arr = np.array(arr_list)
        mask = np.random.rand(len(arr)) < p_drop
        
        random_words = np.random.randint(size=arr.shape, high=1640,low=9)
        arr = np.where(mask,random_words,arr)

        new_list = list(arr)
        new_str = ' '.join(str(x) for x in new_list)
        return new_str
    def data_corruption(self,my_str):
        arr_list = self.str2list(my_str)
        arr_list = self.text_infilling(arr_list)
        arr_list = self.sentence_shuffle(arr_list)
        arr_str = self.list2str(arr_list)
        return arr_str
    
    def list2str(self,my_list):
        my_str = ' '.join(str(x) for x in my_list)
        return my_str
    def str2list(self,mystr):
        mylist = [x for x in mystr.split()]
        return mylist

class DAEdataset_DC(BaseDataset):
    def __init__(self, data_file_list, input_l, output_l,tokenizer):
        if not isinstance(data_file_list,list):
            data_file_list =[data_file_list]   
        self.samples = []
        for data_file in data_file_list:
            self.samples.extend([line for line in open(data_file, 'r',encoding='utf-8').readlines()])
        self.input_l = input_l
        self.output_l = output_l
        self.tokenizer = tokenizer
        self.mask_token = tokenizer.mask_token_id
    def __len__(self):
        return len(self.samples)
    def _try_getitem(self, idx):
        description = self.samples[idx]
        source = description
        description = self.tokenizer(description,max_length=64)['input_ids'][1:-1]
        source_corrupted = self.tokenizer.decode(self.data_corruption(description)).replace(' ','')
        source_noisy = self.tokenizer.decode(self.token_replace(description,p=0.8)).replace(' ','')
        return source_corrupted.strip(),source.strip(),source.strip()
    
    def text_infilling(self,my_list,span_ratio=0.15):
        #text infill
        #span_start 和 span_end 之间的text会被mask,会保留end
        temp_list = my_list.copy()
        arr_list = temp_list
        span_max_length = int(span_ratio*len(arr_list))
        span_total_length = 0
        arr_list_span = []
        span_end = 0
        while span_total_length < span_max_length and len(arr_list)>1:
            span_start = random.randint(0,(len(arr_list)-1)//2)
            # print(span_start)
            # sys.exit(1)
            span_length =  np.random.poisson(lam=3)
            span_length = min(span_length,span_max_length-span_total_length)
            span_end = min(span_start+span_length,len(arr_list)-1)
            if random.random()<0.1:
                arr_list_span = arr_list_span + arr_list[0:span_start]+[random.randint(103,self.tokenizer.vocab_size)]
            else:
                arr_list_span = arr_list_span + arr_list[0:span_start]+[self.mask_token]
            arr_list = temp_list[span_end:]
            span_total_length += span_length

        arr_list_span += arr_list
        return arr_list_span
    
    def sentence_shuffle(self,my_list):
        arr_list = my_list.copy()
        sentence_list=[]
        sentence_end = []
        item=[]
        for i in range(len(arr_list)):
            item.append(arr_list[i])
            if arr_list[i] in sentence_end or i==len(arr_list)-1:
                sentence_list.append(item.copy())
                item = []
        random.shuffle(sentence_list)
        arr_list_shuffle = []
        for k in sentence_list:
            arr_list_shuffle.extend(k)
        return arr_list_shuffle

    def token_replace(self,arr_list,p=1.0,p_drop=0.15):
        #text infill
        if random.random()>p:
            return arr_list

        #drop as BERT
        arr = np.array(arr_list)
        mask = np.random.rand(len(arr)) < p_drop
        
        random_words = np.random.randint(size=arr.shape, high=self.tokenizer.vocab_size,low=103)
        arr = np.where(mask,random_words,arr)

        new_list = list(arr)
        # new_str = ''.join(str(x) for x in new_list)
        return new_list

    def data_corruption(self,arr_list):
        arr_list = self.text_infilling(arr_list)
        arr_list = self.sentence_shuffle(arr_list)
        return arr_list
    
    def list2str(self,my_list):
        my_str = ''.join(str(x) for x in my_list)
        return my_str
    def str2list(self,mystr):
        return list(mystr)

if __name__== '__main__':
    import sys
    tokenizer = BertTokenizer.from_pretrained('d:/download/')
    dc_dataset = DAEdataset_DC('product_names.txt',64,64,tokenizer=tokenizer)
    print(dc_dataset._try_getitem(10))
    # for i in range(10):
    #     print(dc_dataset.data_corruption('轮胎性能测试,这是一个测试的例子呀'))
    # sys.exit(1)
