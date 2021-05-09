#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import copy
import time
import json
import random
import pickle
import argparse
import numpy as np

from konlpy.tag import Mecab
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F

from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from datasets import load_dataset, load_from_disk
from transformers import (BertConfig,
                          BertModel,
                          BertPreTrainedModel,
                          AdamW,
                          TrainingArguments,
                          get_linear_schedule_with_warmup,
                          set_seed,
                          AutoTokenizer)

from elasticsearch_retrieval import elastic_setting, elastic_retrieval

# In[2]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    set_seed(seed)
    
seed_everything(seed=2021)

def get_pickle(pickle_path):
    f = open(pickle_path, "rb")
    dataset = pickle.load(f)
    f.close()
    return dataset

training_dataset = get_pickle("/opt/ml/new_dataset/preprocess_train.pkl")
validation_dataset = get_pickle("/opt/ml/new_dataset/preprocess_valid.pkl")

# In[7]:


model_checkpoint = 'bert-base-multilingual-cased'
p_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
p_tokenizer.model_max_length = 1536
q_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# In[19]:


class TrainRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, p_tokenizer, q_tokenizer):
        self.dataset = dataset
        self.p_tokenizer = p_tokenizer
        self.q_tokenizer = q_tokenizer
        
    def __getitem__(self, idx):
        question = self.dataset['question'][idx]
        top20 = self.dataset['top20'][idx]
        target = self.dataset['answer_idx'][idx]
        
        p_seqs = self.p_tokenizer(top20,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')
        q_seqs = self.q_tokenizer(question,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')
        
        p_input_ids = p_seqs['input_ids']
        p_attention_mask = p_seqs['attention_mask']
        p_token_type_ids = p_seqs['token_type_ids']
        
        q_input_ids = q_seqs['input_ids']
        q_attention_mask = q_seqs['attention_mask']
        q_token_type_ids = q_seqs['token_type_ids']
        
        p_input_ids_list = torch.Tensor([])
        p_attention_mask_list = torch.Tensor([])
        p_token_type_ids_list = torch.Tensor([])
        for i in range(len(p_attention_mask)):
            str_idx, end_idx = self._select_range(p_attention_mask[i])

            p_input_ids_tmp = torch.cat([torch.Tensor([101]), p_input_ids[i][str_idx:end_idx], torch.Tensor([102])]).int().long()
            p_attention_mask_tmp = p_attention_mask[i][str_idx-1:end_idx+1].int().long()
            p_token_type_ids_tmp = p_token_type_ids[i][str_idx-1:end_idx+1].int().long()

            p_input_ids_list = torch.cat([p_input_ids_list, p_input_ids_tmp.unsqueeze(0)]).int().long()
            p_attention_mask_list = torch.cat([p_attention_mask_list, p_attention_mask_tmp.unsqueeze(0)]).int().long()
            p_token_type_ids_list = torch.cat([p_token_type_ids_list, p_token_type_ids_tmp.unsqueeze(0)]).int().long()
            
        return p_input_ids_list, p_attention_mask_list, p_token_type_ids_list, q_input_ids, q_attention_mask, q_token_type_ids, target

    def __len__(self):
        return len(self.dataset['question'])

    def _select_range(self, attention_mask):
        sent_len = len([i for i in attention_mask if i != 0])
        if sent_len <= 512:
            return 1, 511
        else:
            start_idx = random.randint(1, sent_len-511)
            end_idx = start_idx + 510
            return start_idx, end_idx

class TestRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, p_tokenizer, q_tokenizer):
        self.dataset = dataset
        self.p_tokenizer = p_tokenizer
        self.q_tokenizer = q_tokenizer
        
    def __getitem__(self, idx):
        question = self.dataset['question'][idx]
        top20 = self.dataset['top20'][idx]
        target = self.dataset['answer_idx'][idx]
        
        p_seqs = self.p_tokenizer(top20,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')
        q_seqs = self.q_tokenizer(question,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')
        
        p_input_ids = p_seqs['input_ids']
        p_attention_mask = p_seqs['attention_mask']
        p_token_type_ids = p_seqs['token_type_ids']
        
        q_input_ids = q_seqs['input_ids']
        q_attention_mask = q_seqs['attention_mask']
        q_token_type_ids = q_seqs['token_type_ids']
        
        p_input_ids_list = torch.Tensor([])
        p_attention_mask_list = torch.Tensor([])
        p_token_type_ids_list = torch.Tensor([])
        for i in range(len(p_attention_mask)):
            ids_list = self._select_range(p_attention_mask[i])
            if i == target:
                target = list(range(len(p_input_ids_list), len(p_input_ids_list)+len(ids_list)))
            for str_idx, end_idx in ids_list:
                p_input_ids_tmp = torch.cat([torch.Tensor([101]), p_input_ids[i][str_idx:end_idx], torch.Tensor([102])]).int().long()
                p_attention_mask_tmp = p_attention_mask[i][str_idx-1:end_idx+1].int().long()
                p_token_type_ids_tmp = p_token_type_ids[i][str_idx-1:end_idx+1].int().long()

                p_input_ids_list = torch.cat([p_input_ids_list, p_input_ids_tmp.unsqueeze(0)]).int().long()
                p_attention_mask_list = torch.cat([p_attention_mask_list, p_attention_mask_tmp.unsqueeze(0)]).int().long()
                p_token_type_ids_list = torch.cat([p_token_type_ids_list, p_token_type_ids_tmp.unsqueeze(0)]).int().long()

        return p_input_ids_list, p_attention_mask_list, p_token_type_ids_list, q_input_ids, q_attention_mask, q_token_type_ids, target

    def __len__(self):
        return len(self.dataset['question'])
    
    def _select_range(self, attention_mask):
        sent_len = len([i for i in attention_mask if i != 0])
        if sent_len <= 512:
            return [(1,511)]
        else:
            num = sent_len // 255
            res = sent_len % 255
            if res == 0:
                num -= 1
            ids_list = []
            for n in range(num):
                if res > 0 and n == num-1:
                    end_idx = sent_len-1
                    start_idx = end_idx - 510
                else:
                    start_idx = n*255+1
                    end_idx = start_idx + 510
                ids_list.append((start_idx, end_idx))
            return ids_list
# In[20]:


train_dataset = TrainRetrievalDataset(training_dataset, p_tokenizer, q_tokenizer)
valid_dataset = TestRetrievalDataset(validation_dataset, p_tokenizer, q_tokenizer)


# In[21]:


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids)
        pooled_output = outputs[1]
        return pooled_output


# In[22]:


p_encoder = BertEncoder.from_pretrained(model_checkpoint)
q_encoder = BertEncoder.from_pretrained(model_checkpoint)

if torch.cuda.is_available():
    p_encoder.to('cuda')
    q_encoder.to('cuda')
    print('GPU enabled')


# In[23]:


args = TrainingArguments(output_dir='result/dense_retrieval',
                         evaluation_strategy='epoch',
                         learning_rate=1e-5,
                         per_device_train_batch_size=16,
                         per_device_eval_batch_size=1,
                         gradient_accumulation_steps=1,
                         num_train_epochs=10,
                         weight_decay=0.01)


# In[24]:


# Dataloader
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              batch_size=args.per_device_train_batch_size)

valid_sampler = RandomSampler(valid_dataset)
valid_dataloader = DataLoader(valid_dataset,
                              sampler=valid_sampler,
                              batch_size=args.per_device_eval_batch_size)


# In[25]:


# Optimizer
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [{'params': [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                                {'params': [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
                                {'params': [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                                {'params': [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
                                ]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
scaler = GradScaler()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
criterion = nn.NLLLoss()


# In[26]:


# -- logging
log_dir = os.path.join(args.output_dir)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
else:
    raise
logger = SummaryWriter(log_dir=log_dir)


# In[27]:


# Start training!
best_loss = 1e9
best_acc = 0.0
global_step = 0

train_iterator = trange(int(args.num_train_epochs), desc='Epoch')
for epoch in train_iterator:
    optimizer.zero_grad()
    p_encoder.zero_grad()
    q_encoder.zero_grad()
    
    ## train
    epoch_iterator = tqdm(train_dataloader, desc="train Iteration")
    p_encoder.to('cuda').train()
    q_encoder.to('cuda').train()

    running_loss, running_acc, num_cnt = 0, 0, 0
    with torch.set_grad_enabled(True):
        for step, batch_list in enumerate(epoch_iterator):
            p_input_ids = batch_list[0]
            p_attention_mask = batch_list[1]
            p_token_type_ids = batch_list[2]
            q_input_ids = batch_list[3]
            q_attention_mask = batch_list[4]
            q_token_type_ids = batch_list[5]
            targets_batch = batch_list[6]
            
            for i in range(args.per_device_train_batch_size):
                batch = (p_input_ids[i],
                         p_attention_mask[i],
                         p_token_type_ids[i],
                         q_input_ids[i],
                         q_attention_mask[i],
                         q_token_type_ids[i])
                
                targets = torch.tensor([targets_batch[i]]).long()

#                 batch = tuple(t.squeeze(0) if i < 6 else t for i, t in enumerate(batch))
                batch = tuple(t.to('cuda') for t in batch)

                p_inputs = {'input_ids' : batch[0],
                            'attention_mask' : batch[1],
                            'token_type_ids': batch[2]}
                p_outputs = p_encoder(**p_inputs)     # (20, E)

                q_inputs = {'input_ids' : batch[3],
                            'attention_mask' : batch[4],
                            'token_type_ids': batch[5]}
                q_outputs = q_encoder(**q_inputs)     # (1, E)

                # Calculate similarity score & loss
                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1)) # (1, E) x (E, N) = (1, 20)
                # target : position of positive samples = diagonal element
                if torch.cuda.is_available():
                    targets = targets.to('cuda')
                sim_scores = F.log_softmax(sim_scores, dim=1)
                _, preds = torch.max(sim_scores, 1)

                loss = criterion(sim_scores, targets)
                scaler.scale(loss).backward()
            
            if (step+1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                p_encoder.zero_grad()
                q_encoder.zero_grad()

                logger.add_scalar(f"Train/loss", loss.item(), epoch*len(epoch_iterator) + step)
                logger.add_scalar(f"Train/accuracy", torch.sum(preds.cpu() == targets.cpu()).item()/len(batch[-1])*100, epoch*len(epoch_iterator) + step)
                global_step += 1

            running_loss += loss.item()
            running_acc += torch.sum(preds.cpu() == targets.cpu())
            num_cnt += 1
    epoch_loss = float(running_loss / num_cnt)
    epoch_acc  = float((running_acc.double() / num_cnt).cpu()*100)
    print(f'global step-{global_step} | Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}')

    ## valid
    epoch_iterator = tqdm(valid_dataloader, desc="valid Iteration")
    p_encoder.to('cuda').eval()
    q_encoder.to('cuda').eval()

    running_loss, running_acc, num_cnt = 0, 0, 0
    for step, batch in enumerate(epoch_iterator):
        with torch.set_grad_enabled(False):
            batch = tuple(t.squeeze(0) if i < 6 else t for i, t in enumerate(batch))
            targets = batch[-1]
            if torch.cuda.is_available():
                batch = tuple(t.to('cuda') for t in batch[:-1])

            p_inputs = {'input_ids' : batch[0],
                        'attention_mask' : batch[1],
                        'token_type_ids': batch[2]}
            p_outputs = p_encoder(**p_inputs)     # (N, E)

            q_inputs = {'input_ids' : batch[3],
                        'attention_mask' : batch[4],
                        'token_type_ids': batch[5]}
            q_outputs = q_encoder(**q_inputs)     # (1, E)

            # Calculate similarity score & loss
            sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1)) # (1, E) x (E, N) = (1, N)
            sim_scores = F.log_softmax(sim_scores, dim=1)
            _, preds = torch.max(sim_scores, 1)

            if preds.item() in targets:
                running_acc += 1
            num_cnt += 1

    epoch_acc  = float((running_acc / num_cnt)*100)
    logger.add_scalar(f"Val/accuracy", epoch_acc, epoch)
    print(f'global step-{global_step} | Accuracy: {epoch_acc:.2f}')

    if epoch_acc > best_acc:
        best_idx = epoch
        best_acc = epoch_acc
        best_p_model_wts = copy.deepcopy(p_encoder.cpu().state_dict())
        best_q_model_wts = copy.deepcopy(q_encoder.cpu().state_dict())
        print(f'\t==> best model saved - {best_idx} / Accuracy: {best_acc:.2f}')

        save_path = os.path.join(args.output_dir, 'model')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(p_encoder.cpu().state_dict(), os.path.join(save_path, 'best_p_model.pt'))
        torch.save(q_encoder.cpu().state_dict(), os.path.join(save_path, 'best_q_model.pt'))
        print('model saved !!')
