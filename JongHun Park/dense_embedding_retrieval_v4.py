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
                          TrainingArguments,
                          AutoTokenizer)


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


# In[3]:


org_dataset = load_from_disk("/opt/ml/input/data/data/train_dataset")
print("*"*40, "query dataset", "*"*40)
print(org_dataset)


# In[4]:


training_dataset = org_dataset['train']
validation_dataset = org_dataset['validation']


# In[5]:


mecab = Mecab()
def morphs_split(text):
    text = mecab.morphs(text)
    return ' '.join(text)

def context_split(text):
    text = ' '.join(text.strip().split('\\n')).strip()
    sent_list = text.strip().split('. ')
    text = ''
    for sent in sent_list:
        sent = mecab.morphs(sent)
        text += ' '.join(sent)+'[SEP]'
    return text[:-5]

def sentence_split(text):
    text_list = [sent for sent in map(lambda x : x.strip(), text.split('[SEP]')) if sent != '']
    return text_list


# In[7]:


model_checkpoint = 'bert-base-multilingual-cased'
p_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
p_tokenizer.model_max_length = 1536
q_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# In[9]:


def preprocessing(dataset):
    q_test = dataset['question']
    p_test = dataset['context']
    a_test = dataset['answers']

    # idx = 45
    new_dataset = {'question':[], 'context':[], 'answers':[], 'position_ids':torch.Tensor([]).int()}
    for idx in range(len(p_test)):
        question = q_test[idx]
        new_question = morphs_split(question)

        context = p_test[idx]
        start_idx = a_test[idx]['answer_start'][0]
        answer = a_test[idx]['text'][0]

        front_context = context_split(context[:start_idx])
        back_context = context_split(context[start_idx+len(answer):])
        new_answer = morphs_split(answer)

        new_context = ' '.join([front_context, new_answer, back_context])

        pos_ids = torch.Tensor([])
        for sentence in sentence_split(new_context):
            sent_token = p_tokenizer(sentence)
            sent_token_len = len(sent_token['input_ids'][1:-1])
            if sent_token_len < 511:
                sent_pos_ids = torch.arange(0, sent_token_len+1)[1:]
            else:
                sent_pos_ids = torch.arange(0, 511)[1:]
                sent_pos_ids = torch.cat([sent_pos_ids, torch.ones(sent_token_len-510)*510])
            pos_ids = torch.cat([pos_ids, sent_pos_ids])
        pos_ids = torch.cat([torch.Tensor([0]).int(), pos_ids, torch.Tensor([511]).int()])
        add_tokens = torch.zeros(p_tokenizer.model_max_length - pos_ids.size()[0])
        pos_ids = torch.cat([pos_ids, add_tokens]).int()
        new_context = ' '.join(sentence_split(new_context))
        
        new_dataset['question'].append(new_question)
        new_dataset['context'].append(new_context)
        new_dataset['answers'].append(new_answer)
        new_dataset['position_ids'] = torch.cat([new_dataset['position_ids'], pos_ids.unsqueeze(0)]).int()
    return new_dataset


# In[10]:


new_training_dataset = preprocessing(training_dataset)
new_validation_dataset = preprocessing(validation_dataset)

# In[17]:


tr_p_seqs = p_tokenizer(new_training_dataset['context'],
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt')
tr_q_seqs = q_tokenizer(new_training_dataset['question'],
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt')
va_p_seqs = p_tokenizer(new_validation_dataset['context'],
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt')
va_q_seqs = q_tokenizer(new_validation_dataset['question'],
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt')


# In[19]:


class TrainRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, p_seqs, q_seqs, dataset):
        self.p_seqs = p_seqs
        self.q_seqs = q_seqs
        self.dataset = dataset
        
    def __getitem__(self, idx):
        p_input_ids = self.p_seqs['input_ids'][idx]
        p_attention_mask = self.p_seqs['attention_mask'][idx]
        p_token_type_ids = self.p_seqs['token_type_ids'][idx]
        p_position_ids = self.dataset['position_ids'][idx]
        q_input_ids = self.q_seqs['input_ids'][idx]
        q_attention_mask = self.q_seqs['attention_mask'][idx]
        q_token_type_ids = self.q_seqs['token_type_ids'][idx]
        
        str_idx, end_idx, add_token = self._select_range(p_attention_mask)
        if add_token:
            p_input_ids = torch.cat([torch.Tensor([101]), p_input_ids[str_idx:end_idx], torch.Tensor([102])]).int().long()
            p_attention_mask = p_attention_mask[str_idx-1:end_idx+1].int().long()
            p_token_type_ids = p_token_type_ids[str_idx-1:end_idx+1].int().long()
            p_position_ids = torch.cat([torch.Tensor([0]), p_position_ids[str_idx:end_idx], torch.Tensor([511])]).int().long()
        else:
            p_input_ids = p_input_ids[str_idx:end_idx].int().long()
            p_attention_mask = p_attention_mask[str_idx:end_idx].int().long()
            p_token_type_ids = p_token_type_ids[str_idx:end_idx].int().long()
            p_position_ids = p_position_ids[str_idx:end_idx].int().long()
            
        return p_input_ids, p_attention_mask, p_token_type_ids, p_position_ids, q_input_ids, q_attention_mask, q_token_type_ids

    def __len__(self):
        return len(self.p_seqs['input_ids'])

    def _select_range(self, attention_mask):
        sent_len = len([i for i in attention_mask if i != 0])
        if sent_len <= 512:
            return 0, 512, False
        else:
            start_idx = random.randint(1, sent_len-511)
            end_idx = start_idx + 510
            return start_idx, end_idx, True

class TestRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, p_seqs, q_seqs, dataset):
        self.p_seqs = p_seqs
        self.q_seqs = q_seqs
        self.dataset = dataset
        
    def __getitem__(self, idx):
        p_input_ids = self.p_seqs['input_ids'][idx]
        p_attention_mask = self.p_seqs['attention_mask'][idx]
        p_token_type_ids = self.p_seqs['token_type_ids'][idx]
        p_position_ids = self.dataset['position_ids'][idx]
        q_input_ids = self.q_seqs['input_ids'][idx]
        q_attention_mask = self.q_seqs['attention_mask'][idx]
        q_token_type_ids = self.q_seqs['token_type_ids'][idx]
        
        ids_list, add_token = self._select_range(p_attention_mask)
        if add_token:
            p_input_ids_list = torch.Tensor([])
            p_attention_mask_list = torch.Tensor([])
            p_token_type_ids_list = torch.Tensor([])
            p_position_ids_list = torch.Tensor([])
            for str_idx, end_idx in ids_list:
                p_input_ids_list = torch.cat([p_input_ids_list, torch.cat([torch.Tensor([101]), p_input_ids[str_idx:end_idx], torch.Tensor([102])]).unsqueeze(0)]).int().long()
                p_attention_mask_list = torch.cat([p_attention_mask_list, p_attention_mask[str_idx-1:end_idx+1].unsqueeze(0)]).int().long()
                p_token_type_ids_list = torch.cat([p_token_type_ids_list, p_token_type_ids[str_idx-1:end_idx+1].unsqueeze(0)]).int().long()
                p_position_ids_list = torch.cat([p_position_ids_list, torch.cat([torch.Tensor([0]), p_position_ids[str_idx:end_idx], torch.Tensor([511])]).unsqueeze(0)]).int().long()
        else:
            str_idx, end_idx = ids_list[0]
            p_input_ids_list = p_input_ids[str_idx:end_idx].unsqueeze(0).int().long()
            p_attention_mask_list = p_attention_mask[str_idx:end_idx].unsqueeze(0).int().long()
            p_token_type_ids_list = p_token_type_ids[str_idx:end_idx].unsqueeze(0).int().long()
            p_position_ids_list = p_position_ids[str_idx:end_idx].unsqueeze(0).int().long()
            
        return p_input_ids_list, p_attention_mask_list, p_token_type_ids_list, p_position_ids_list, q_input_ids, q_attention_mask, q_token_type_ids

    def __len__(self):
        return len(self.p_seqs['input_ids'])
    
    def _select_range(self, attention_mask):
        sent_len = len([i for i in attention_mask if i != 0])
        if sent_len <= 512:
            return [(0,512)], False
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
            return ids_list, True
# In[20]:


train_dataset = TrainRetrievalDataset(tr_p_seqs, tr_q_seqs, new_training_dataset)
valid_dataset = TestRetrievalDataset(va_p_seqs, va_q_seqs, new_validation_dataset)


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
                         num_train_epochs=30,
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
    
    for phase in ['train', 'valid']:
        if phase == 'train':
            epoch_iterator = tqdm(train_dataloader, desc=f"{phase} Iteration")
            p_encoder.to('cuda').train()
            q_encoder.to('cuda').train()
        else:
            epoch_iterator = tqdm(valid_dataloader, desc=f"{phase} Iteration")
            p_encoder.to('cuda').eval()
            q_encoder.to('cuda').eval()
            
            p_outputs_list = torch.Tensor([])
            q_outputs_list = torch.Tensor([])
            
        running_loss, running_acc, num_cnt = 0, 0, 0
        for step, batch in enumerate(epoch_iterator):
            with torch.set_grad_enabled(phase == 'train'):
                if phase == 'valid':
                    batch = tuple(t.squeeze(0) if i < 4 else t for i, t in enumerate(batch))
                
                if torch.cuda.is_available():
                    batch = tuple(t.to('cuda') for t in batch)
                    
                p_inputs = {'input_ids' : batch[0],
                            'attention_mask' : batch[1],
                            'token_type_ids': batch[2],
                            'position_ids': batch[3]}
                q_inputs = {'input_ids' : batch[4],
                            'attention_mask' : batch[5],
                            'token_type_ids': batch[6]}
                
                p_outputs = p_encoder(**p_inputs)     # (B, E)
                q_outputs = q_encoder(**q_inputs)     # (B, E)
                
                if phase == 'valid':
                    p_output = F.avg_pool1d(torch.transpose(p_outputs.cpu().unsqueeze(0), 1, 2), kernel_size=len(p_outputs), stride=1).cpu().squeeze(0)   # (E, 1)
                    p_outputs_list = torch.cat([p_outputs_list, p_output.cpu()], dim=1).cpu()
                    q_outputs_list = torch.cat([q_outputs_list, q_outputs.cpu()], dim=0).cpu()
                    continue
                
                # Calculate similarity score & loss
                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1)) # (B, E) x (E, B) = (B, B)
                # target : position of positive samples = diagonal element
                targets = torch.arange(0, args.per_device_train_batch_size).long()
                if torch.cuda.is_available():
                    targets = targets.to('cuda')
                sim_scores = F.log_softmax(sim_scores, dim=1)
                _, preds = torch.max(sim_scores, 1)
                
                loss = criterion(sim_scores, targets)
                if phase == 'train':
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
                
                running_loss += loss.item() * len(batch[-1])
                running_acc += torch.sum(preds.cpu() == targets.cpu())
                num_cnt += len(batch[-1])
        
        if phase == 'valid':
            with torch.set_grad_enabled(phase == 'train'):
            # Calculate similarity score & loss
                sim_scores = torch.matmul(q_outputs_list.cpu(), p_outputs_list.cpu()).cpu() # (B, E) x (E, B) = (B, B)
                # target : position of positive samples = diagonal element
                targets = torch.arange(0, len(q_outputs_list.cpu())).long().cpu()
                sim_scores = F.log_softmax(sim_scores.cpu(), dim=1).cpu()
                _, preds = torch.max(sim_scores, 1)
                
                loss = criterion(sim_scores.cpu(), targets.cpu()).cpu()
                
                running_loss += loss.item() * len(q_outputs_list.cpu())
                running_acc += torch.sum(preds.cpu() == targets.cpu())
                num_cnt += len(q_outputs_list.cpu())
        
        epoch_loss = float(running_loss / num_cnt)
        epoch_acc  = float((running_acc.double() / num_cnt).cpu()*100)
            
        if phase == 'valid':
            logger.add_scalar(f"Val/loss", epoch_loss, epoch)
            logger.add_scalar(f"Val/accuracy", epoch_acc, epoch)
        print(f'global step-{global_step} | Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}')
        
        if phase == 'valid' and epoch_acc > best_acc:
            best_idx = epoch
            best_loss = epoch_loss
            best_acc = epoch_acc
            best_p_model_wts = copy.deepcopy(p_encoder.cpu().state_dict())
            best_q_model_wts = copy.deepcopy(q_encoder.cpu().state_dict())
            print(f'\t==> best model saved - {best_idx} / Loss: {best_loss:.4f}, Accuracy: {best_acc:.2f}')
p_encoder.cpu().load_state_dict(best_p_model_wts)
q_encoder.cpu().load_state_dict(best_q_model_wts)

save_path = os.path.join(args.output_dir, 'model')
if not os.path.exists(save_path):
    os.mkdir(save_path)
torch.save(p_encoder.state_dict(), os.path.join(save_path, 'best_p_model.pt'))
torch.save(q_encoder.state_dict(), os.path.join(save_path, 'best_q_model.pt'))
print('model saved !!')
