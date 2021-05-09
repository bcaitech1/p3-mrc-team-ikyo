import os
import re
import json
import pickle

import pandas as pd
from elasticsearch import Elasticsearch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_metric, load_from_disk, load_dataset, Features, Value, Sequence, DatasetDict, Dataset


def save_pickle(save_path, data_set):
    file = open(save_path, "wb")
    pickle.dump(data_set, file)
    file.close()
    return None


def get_pickle(pickle_path):
    f = open(pickle_path, "rb")
    dataset = pickle.load(f)
    f.close()
    return dataset


def preprocess(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥<>()\s\.\?!》《≪≫\'<>〈〉:‘’%,『』「」＜＞・\"-“”∧]", "", text)
    return text


def run_preprocess(data_dict):
    context = data_dict["context"]
    start_ids = data_dict["answers"]["answer_start"][0]
    before = data_dict["context"][:start_ids]
    after = data_dict["context"][start_ids:]
    process_before = preprocess(before)
    process_after = preprocess(after)
    process_data = process_before + process_after
    ids_move = len(before) - len(process_before)
    data_dict["context"] = process_data
    data_dict["answers"]["answer_start"][0] = start_ids - ids_move
    return data_dict


def search_es(es_obj, index_name, question_text, n_results):
    query = {
            'query': {
                'match': {
                    'document_text': question_text
                    }
                }
            }
    res = es_obj.search(index=index_name, body=query, size=n_results)
    return res


def make_custom_dataset(dataset_path) :
    if not os.path.isdir("/opt/ml/input/data/train_dataset") :
        raise Exception ("Set the data path to '/opt/ml/input/data/.'")
    
    train_f = Features({'answers': Sequence(feature={'text': Value(dtype='string', id=None), 'answer_start': Value(dtype='int32', id=None)}, length=-1, id=None),
                        'context': Value(dtype='string', id=None),
                        'id': Value(dtype='string', id=None),
                        'question': Value(dtype='string', id=None)})

    if not os.path.isfile("/opt/ml/input/data/preprocess_train.pkl") :
        train_data = load_from_disk("/opt/ml/input/data/train_dataset")['train']
        val_data = load_from_disk("/opt/ml/input/data/train_dataset")['validation']
        test_data = load_from_disk("/opt/ml/input/data/test_dataset")['validation']
        
        new_train_data, new_val_data = [], []
        for data in train_data:
            new_data = run_preprocess(data)
            new_train_data.append(new_data)
        for data in val_data:
            new_data = run_preprocess(data)
            new_val_data.append(new_data)
        
        train_df = pd.DataFrame(new_train_data)
        val_df = pd.DataFrame(new_val_data)
        train_datasets = DatasetDict({'train': Dataset.from_pandas(train_df, features=train_f), 'validation': Dataset.from_pandas(val_df, features=train_f)})
        save_pickle("/opt/ml/input/data/preprocess_train.pkl", train_datasets)
        
        if 'preprocess' in dataset_path :
            return train_datasets
    
    if 'squad' in dataset_path :
        train_data = get_pickle("/opt/ml/input/data/preprocess_train.pkl")["train"]
        koquard_dataset = load_dataset("squad_kor_v1")["train"]
        val_data = get_pickle("/opt/ml/input/data/preprocess_train.pkl")["validation"]

        df_train_data = pd.DataFrame(train_data)
        df_korquad_data = pd.DataFrame(koquard_dataset)
        df_val_data = pd.DataFrame(val_data)

        df_train_data = df_train_data[["answers", "context", "id", "question"]]
        df_korquad_data = df_korquad_data[["answers", "context", "id", "question"]]
        df_val_data = df_val_data[["answers", "context", "id", "question"]]
        df_total_train = pd.concat([train_df, koquard_df])

        total_datasets = DatasetDict({'train': Dataset.from_pandas(df_total_train, features=train_f), 'validation': Dataset.from_pandas(df_val_data, features=train_f)})
        save_pickle("/opt/ml/input/data/add_squad_kor_v1_2.pkl", total_datasets)
        
        return total_datasets

    if 'concat' in dataset_path :
        data = get_pickle("/opt/ml/input/data/preprocess_train.pkl")
        train_file = data["train"]
        validation_file = data["validation"]

        train_qa = [{"id" : train_file[i]["id"], "question" : train_file[i]["question"], "answers" : train_file[i]["answers"], "context" : train_file[i]["context"]} for i in range(len(train_file))]
        validation_qa = [{"id" : validation_file[i]["id"], "question" : validation_file[i]["question"], "answers" : validation_file[i]["answers"], "context" : validation_file[i]["context"]} for i in range(len(validation_file))]

        config = {'host':'localhost', 'port':9200}
        es = Elasticsearch([config])

        # train concat데이터셋 만들기
        for step, question in enumerate(train_qa):
            # 5 => k의 수 (뽑아올 수)
            res = search_es(es, "nori-index", question["question"], 5)
            context_list = [(hit['_source']['document_text'], hit['_score']) for hit in res['hits']['hits']]
            add_text = train_qa[step]["context"]
            count = 0
            for context in context_list:
                #같은것이 있을 경우 continue 하여 concat X
                if question["context"] == context[0]:
                    continue
                add_text += " " + context[0]
                count += 1
                if count == 4:
                    break
            train_qa[step]["context"] = add_text

        # validation도 똑같이 사용
        for step, question in enumerate(validation_qa):
            res = search_es(es, "nori-index", question["question"], 5)
            context_list = [(hit['_source']['document_text'], hit['_score']) for hit in res['hits']['hits']]
            add_text = validation_qa[step]["context"]
            count = 0
            for context in context_list:
                if question["context"] == context[0]:
                    continue
                add_text += " " + context[0]
                count += 1
                if count == 4:
                    break
            validation_qa[step]["context"] = add_text
        
        train_df = pd.DataFrame(train_qa)
        val_df = pd.DataFrame(validation_qa)
        train_datasets = DatasetDict({'train': Dataset.from_pandas(train_df, features=train_f), 'validation': Dataset.from_pandas(val_df, features=train_f)})
        save_pickle("/opt/ml/input/data/train_concat5.pkl", train_datasets)
        
        return train_datasets