from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import json
import csv
from datasets import load_dataset

import torch
import random
import warnings
from torch.utils.data.distributed import DistributedSampler

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

def text_expand(text, max_len):
    if text.size(1) >= max_len:
        text = text[:, :max_len]
    else:
        n = text.size(1)
        rep = max_len // n
        res = max_len % n
        
        if res == 0:
            text = text.repeat([1, rep]) 
        else:
            text = torch.concat([text.repeat([1, rep]), text[:, :res]], dim=1)
    return text


class DatasetLoader(Dataset):
    def __init__(self, args, tokenizer, src):
        super().__init__()
        
        self.args = args
        self.tokenizer = tokenizer
        
        if src == 'train':
            data_src = src
            
        elif src == 'validation':
            data_src = src
        
        else:
            data_src = 'validation'
            
            
        if self.args.method == 'piqa':
            question = 'goal'
            ans = ['sol1', 'sol2']
            label = 'label'
            
            dataset = load_dataset(self.args.method)[data_src]
            self.q = dataset[question]
            labels = dataset[label]
            self.a = [dataset[ans[lab]][i] for i, lab in enumerate(labels)]
            
        elif self.args.method == 'boolq':
            question = 'question'
            ans = 'passage'
            label = 'answer'
            
            dataset = load_dataset(self.args.method)[data_src]
            true_ans = np.array(dataset[label]) == True
            self.q = np.array(dataset[question])[true_ans].tolist()
            self.a = np.array(dataset[ans])[true_ans].tolist()
        
        elif self.args.method == 'winogrande':
            question = 'sentence'
            ans = ['option1', 'option2']
            label = 'answer'
            dataset = load_dataset(self.args.method, 'winogrande_xs')[data_src]
            self.q = dataset[question]
            labels = dataset[label]
            self.a = [dataset[ans[int(lab)-1]][i] for i, lab in enumerate(labels)]
        
        elif self.args.method == 'arc-e':
            question = 'question'
            ans = 'choices'
            label = 'answerKey'
            dataset = load_dataset('ai2_arc', 'ARC-Easy')[data_src]
            self.q = dataset[question]
            labels = dataset[label]
            self.a = [dataset[ans][i]['text'][dataset[ans][i]['label'].index(lab)] \
                      for i, lab in enumerate(labels)]
        
        elif self.args.method == 'arc-c':
            question = 'question'
            ans = 'choices'
            label = 'answerKey'
            dataset = load_dataset('ai2_arc', 'ARC-Challenge')[data_src]
            self.q = dataset[question]
            labels = dataset[label]
            self.a = [dataset[ans][i]['text'][dataset[ans][i]['label'].index(lab)] \
                      for i, lab in enumerate(labels)]
            
        self.n_data = len(self.q)
        
        
        if src == 'validation':
            self.n_data = int(self.n_data*0.8)
            self.q = self.q[:self.n_data]
            self.a = self.a[:self.n_data]
        
        elif src == 'test':
            self.n_data = int(self.n_data*0.2)
            self.q = self.q[-self.n_data:]
            self.a = self.a[-self.n_data:]

        
    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        
        q = self.q[idx]
        a = self.a[idx]
        
        if self.args.method == 'winogrande':
            text = q.replace('_', a) + '\n'
        else:
            text = q + ' ' + a + '\n'
     
        true_text = self.tokenizer(text, return_tensors='pt').input_ids
        true_text = text_expand(true_text, self.args.max_seq_len)
        
        return true_text[0]
    

class Data:
    def __init__(self, args, tokenizer):
        
        if args.test_only != 'True':
            self.train_dataset = DatasetLoader(args, tokenizer, 'train')
       
            self.loader_train = DataLoader(
                        self.train_dataset, 
                        batch_size=args.train_batch_size, shuffle=True, 
                        num_workers=2
                        )
            
            val_dataset = DatasetLoader(args, tokenizer, 'validation')
       
            self.loader_validation = DataLoader(
                        val_dataset, 
                        batch_size=args.eval_batch_size, shuffle=False, 
                        num_workers=2
                        )
        
        test_dataset = DatasetLoader(args, tokenizer, 'test')
   
        self.loader_test = DataLoader(
                    test_dataset, 
                    batch_size=args.eval_batch_size, shuffle=False, 
                    num_workers=2
                    )
      
 