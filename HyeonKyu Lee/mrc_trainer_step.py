import logging
import argparse
import time

import torch
import random
import numpy as np
import pandas as pd
import os
import pickle
import wandb
import math

from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_metric, load_from_disk
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer, AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from data_processing import DataProcessor
from trainer_qa import QuestionAnsweringTrainer
from utils_qa import postprocess_qa_predictions
#from retrieval import SparseRetrieval
from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

logger = logging.getLogger(__name__)

# config 설정
def get_config():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args

# seed 설정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    set_seed(seed)

# 모델, tokenizer, config 가져오기
def get_model(model_args, training_args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True
    )

    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path, 
        config = model_config
    )
    
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)

    return tokenizer, model_config, model, optimizer, scaler, scheduler 

def get_pickle(pickle_path):
    f = open(pickle_path, "rb")
    dataset = pickle.load(f)
    f.close()
    return dataset

def get_data(training_args, tokenizer, text_data_path = "/opt/ml/input/data/data/train_dataset"):
    # squard를 추가한 데이터
    add_squad_ko = False
    # 전처리를 수행한 데이터 
    preprocessing = True
    # concat한 데이터
    concat = False

    if preprocessing:
        text_data = get_pickle("/opt/ml/lastcode/dataset/preprocess_train.pkl")
    elif concat:
        text_data = get_pickle("/opt/ml/lastcode/dataset/train_concat7.pkl")
    else:
        text_data = load_from_disk(text_data_path)


    train_column_names = text_data["train"].column_names
    test_column_names = text_data["validation"].column_names
    data_collator = (
        DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )
    data_processor = DataProcessor(tokenizer)

    # 참일경우 squard 데이터 traind에 추가
    if add_squad_ko:
        train_text = get_pickle("/opt/ml/lastcode/dataset/add_squad_kor_v1_2.pkl")["train"]
        train_column_names = train_text.column_names
    else:
        train_text = text_data["train"]
    val_text = text_data["validation"]

    # 데이터 tokenize,  MRC 모델에 들어갈 수 있도록
    train_dataset = data_processor.train_tokenizer(train_text, train_column_names)
    val_dataset = data_processor.val_tokenzier(val_text, test_column_names)

    train_iter = DataLoader(train_dataset, collate_fn = data_collator, batch_size=training_args.per_device_train_batch_size)
    val_iter = DataLoader(val_dataset, collate_fn = data_collator, batch_size=training_args.per_device_eval_batch_size)

    return text_data, train_iter, val_iter, train_dataset, val_dataset
 
# baseline과 같으니 생략
def post_processing_function(examples, features, predictions, text_data, data_args, training_args):
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        max_answer_length=data_args.max_answer_length,
        output_dir=training_args.output_dir,
    )

    formatted_predictions = [
        {"id": k, "prediction_text": v} for k, v in predictions.items()
    ]
    if training_args.do_predict:
        return formatted_predictions

    elif training_args.do_eval:
        references = [
            {"id": ex["id"], "answers": ex["answers"]}
            for ex in text_data["validation"]
        ]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

# baseline과 같으니 생략
def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    step = 0

    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)

    for i, output_logit in enumerate(start_or_end_logits):
        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat


def training_per_step(model, optimizer, scaler, batch, model_args, data_args, training_args, tokenizer, device):
    model.train()
    with autocast():
        mask_props = 0.8
        mask_p = random.random()
        if mask_p < mask_props:
            # 확률 안에 들면 mask 적용
            batch = custom_to_mask(batch, tokenizer)

        batch = batch.to(device)
        outputs = model(**batch)

        # output안에 loss가 들어있는 형태
        loss = outputs.loss
        scaler.scale(loss).backward()

        # loss 계산
        running_loss += loss.item()*training_args.per_device_train_batch_size
        sample_num += training_args.per_device_train_batch_size

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return running_loss/sample_num


def validating_per_steps(epoch, model, text_data, test_loader, test_dataset, model_args, data_args, training_args, device):
    metric = load_metric("squad")
    if "xlm" in model_args.model_name_or_path:
        test_dataset.set_format(type="torch", columns=["attention_mask", "input_ids"])
    else:
        test_dataset.set_format(type="torch", columns=["attention_mask", "input_ids", "token_type_ids"])

    model.eval()
    all_start_logits = []
    all_end_logits = []

    for batch in test_loader :
        batch = batch.to(device)
        outputs = model(**batch)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        all_start_logits.append(start_logits.detach().cpu().numpy())
        all_end_logits.append(end_logits.detach().cpu().numpy())
    
    max_len = max(x.shape[1] for x in all_start_logits)

    start_logits_concat = create_and_fill_np_array(all_start_logits, test_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, test_dataset, max_len)

    del all_start_logits
    del all_end_logits
    
    test_dataset.set_format(type=None, columns=list(test_dataset.features.keys()))
    output_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(text_data["validation"], test_dataset, output_numpy, text_data, data_args, training_args)
    val_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)

    return val_metric


def custom_to_mask(batch, tokenizer):
    # mask 적용
    mask_token = tokenizer.mask_token_id
    
    for i in range(len(batch["input_ids"])):
        # sep 토큰으로 question과 context가 나뉘어져 있다.
        sep_idx = np.where(batch["input_ids"][i].numpy() == tokenizer.sep_token_id)
        # q_ids = > 첫번째 sep 토큰위치
        q_ids = sep_idx[0][0]
        mask_idxs = set()
        while len(mask_idxs) < 1:
            # 1 ~ q_ids까지가 Question 위치
            ids = random.randrange(1, q_ids)
            mask_idxs.add(ids)

        for mask_idx in list(mask_idxs):
            batch["input_ids"][i][mask_idx] = mask_token
    
    return batch


def train_mrc(model, optimizer, scaler, text_data, train_loader, test_loader, train_dataset, test_dataset, scheduler, model_args, data_args, training_args, tokenizer, device):
    prev_f1 = 0
    global_steps = 0
    for epoch in range(int(training_args.num_train_epochs)):
        running_loss, sample_num = 0, 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
        for step, batch in pbar:
            # training phase
            train_loss = training_per_step(model, optimizer, scaler, batch, model_args, data_args, training_args, tokenizer, device)
            global_steps += 1
            description = f"{epoch}epoch {global_steps: >4d}step | loss: {train_loss: .4f} | best_f1: {prev_f1: .4f}"
            pbar.set_description(description)

            # validating phase
            if global_steps % training_args.logging_steps == 0 :
                with torch.no_grad():
                    val_metric = validating_per_steps(epoch, model, text_data, test_loader, test_dataset, model_args, data_args, training_args, device)
                if val_metric["f1"] > prev_f1:
                    model_name = model_args.model_name_or_path
                    model_name = model_name.split("/")[-1]
                    # backborn 모델의 이름으로 저장 => make submission의 tokenizer부분에 사용하기 위하여
                    torch.save(model, training_args.output_dir + f"/{model_name}.pt")
                    prev_f1 = val_metric["f1"]
                wandb.log({
                'train/loss' : running_loss/sample_num,
                'train/learning_rate' : scheduler.get_last_lr()[0] if scheduler is not None else training_args.learning_rate,
                'eval/exact_match' : val_metric['exact_match'],
                'eval/f1_score' : val_metric['f1'],
                'global_steps': global_steps
                })
                running_loss, sample_num = 0, 0
            else : 
                wandb.log({'global_steps':global_steps})
    
    if scheduler is not None :
        scheduler.step()


def main():
    model_args, data_args, training_args = get_config()
    seed_everything(training_args.seed)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer, model_config, model, optimizer, scaler, scheduler  = get_model(model_args, training_args)
    model.cuda()
    text_data, train_loader, val_loader, train_dataset, val_dataset = get_data(training_args, tokenizer)

    '''wandb setting code'''
    os.environ['WANDB_LOG_MODEL'] = 'true'
    os.environ['WANDB_WATCH'] = 'all'
    os.environ['WANDB_SILENT'] = 'true'
    wandb.login()
    wandb.init(project='P3-MRC', entity='team-ikyo', name=training_args.run_name)
    wandb.watch(model)

    train_mrc(model, optimizer, scaler, text_data, train_loader, val_loader, train_dataset, val_dataset, scheduler, model_args, data_args, training_args, tokenizer, device)

if __name__ == "__main__":
    main()