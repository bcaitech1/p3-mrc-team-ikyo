import logging
import argparse
import time

import torch
import random
import numpy as np
import pandas as pd
import os

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

def get_config():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    set_seed(seed)


def get_model(model_args, training_args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True
    )

    model_config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path, 
        config = model_config
    )
    
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)

    return tokenizer, model_config, model, optimizer, scaler, scheduler 


def get_data(training_args, tokenizer, text_data_path = "/opt/ml/input/data/data/train_dataset"):
    text_data = load_from_disk(text_data_path)
    column_names = text_data["train"].column_names
    data_collator = (
        DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )

    data_processor = DataProcessor(tokenizer)
    train_text = text_data["train"]
    val_text = text_data["validation"]

    train_dataset = data_processor.train_tokenizer(train_text, column_names)
    val_dataset = data_processor.val_tokenzier(val_text, column_names)

    train_iter = DataLoader(train_dataset, collate_fn = data_collator, batch_size=training_args.per_device_train_batch_size)
    val_iter = DataLoader(val_dataset, collate_fn = data_collator, batch_size=training_args.per_device_eval_batch_size)

    return text_data, train_iter, val_iter, train_dataset, val_dataset

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


def test_one_epoch(epcoh, model, text_data, test_loader, test_dataset, model_args, data_args, training_args, device):
    metric = load_metric("squad")
    if "xlm" in model_args.model_name_or_path:
        test_dataset.set_format(type="torch", columns=["attention_mask", "input_ids"])
    else:
        test_dataset.set_format(type="torch", columns=["attention_mask", "input_ids", "token_type_ids"])

    model.eval()

    all_start_logits = []
    all_end_logits = []

    t = time.time()
    loss_sum = 0
    sample_num = 0
    
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), position=0, leave=True)
    for step, batch in pbar:
        batch = batch.to(device)
        outputs = model(**batch)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        all_start_logits.append(start_logits.detach().cpu().numpy())
        all_end_logits.append(end_logits.detach().cpu().numpy())

        loss = outputs.loss
        loss_sum += loss.item()*training_args.per_device_train_batch_size
        sample_num += training_args.per_device_train_batch_size

    max_len = max(x.shape[1] for x in all_start_logits)

    start_logits_concat = create_and_fill_np_array(all_start_logits, test_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, test_dataset, max_len)

    del all_start_logits
    del all_end_logits
    
    test_dataset.set_format(type=None, columns=list(test_dataset.features.keys()))
    output_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(text_data["validation"], test_dataset, output_numpy, text_data, data_args, training_args)
    val_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
    print(f"    validation data : {val_metric}")

    return loss_sum, val_metric

def train_one_epoch(epoch, model, optim, scaler, text_data, train_loader, train_dataset, device, data_args, training_args, tokenizer, scheduler=None):
    model.train()

    t = time.time()
    running_loss = 0
    sample_num = 0      

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
    for step, batch in pbar:
        with autocast():
            batch = batch.to(device)
            outputs = model(**batch)

            loss = outputs.loss
            scaler.scale(loss).backward()

            running_loss += loss.item()*training_args.per_device_train_batch_size
            sample_num += training_args.per_device_train_batch_size

            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

            description = f"epoch {epoch} loss: {running_loss/sample_num: .4f}"
            pbar.set_description(description)

    if scheduler is not None:
        scheduler.step()

    return running_loss, scheduler


def train_mrc(model, optimizer, scaler, text_data, train_loader, test_loader, train_dataset, test_dataset, scheduler, model_args, data_args, training_args, tokenizer, device):
    prev_em = 0
    for epoch in range(int(training_args.num_train_epochs)):
        train_loss, train_scheduler = train_one_epoch(epoch, model, optimizer, scaler, text_data, train_loader,train_dataset, device, data_args, training_args, tokenizer, scheduler)
        with torch.no_grad():
            val_loss, val_metric = test_one_epoch(epoch, model, text_data, test_loader, test_dataset, model_args, data_args, training_args, device)
            if val_metric["exact_match"] > prev_em:
                torch.save(model, training_args.output_dir + f"/{training_args.model_name_or_path}.pt")
        
        '''wandb setting code'''
        wandb.log({
        'train/loss' : train_loss,
        'train/learning_rate' : train_scheduler.get_last_lr()[0] if train_scheduler is not None else training_args.learning_rate,
        'eval/loss' : val_loss,
        'eval/exact_match' : val_metric['exact_match']
        })

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
