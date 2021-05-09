#!/usr/bin/env python
# coding: utf-8

import os
import re
import time
import json
import torch
import pickle
import argparse

from konlpy.tag import Mecab
from tqdm.notebook import tqdm, trange
from transformers import AutoTokenizer
from elasticsearch import Elasticsearch
from subprocess import Popen, PIPE, STDOUT
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_metric, load_from_disk, load_dataset, Features, Value, Sequence, DatasetDict, Dataset

# collapse-hide
def populate_index(es_obj, index_name, evidence_corpus):
    '''
    Loads records into an existing Elasticsearch index

    Args:
        es_obj (elasticsearch.client.Elasticsearch) - Elasticsearch client object
        index_name (str) - Name of index
        evidence_corpus (list) - List of dicts containing data records

    '''

    for i, rec in enumerate(tqdm(evidence_corpus)):
    
        try:
            index_status = es_obj.index(index=index_name, id=i, body=rec)
        except:
            print(f'Unable to load document {i}.')
            
    n_records = es_obj.count(index=index_name)['count']
    print(f'Succesfully loaded {n_records} into {index_name}')
    return

# collapse-hide
def search_es(es_obj, index_name, question_text, n_results):
    '''
    Execute an Elasticsearch query on a specified index
    
    Args:
        es_obj (elasticsearch.client.Elasticsearch) - Elasticsearch client object
        index_name (str) - Name of index to query
        query (dict) - Query DSL
        n_results (int) - Number of results to return
        
    Returns
        res - Elasticsearch response object
    
    '''
    
    # construct query
    query = {
            'query': {
                'match': {
                    'document_text': question_text
                    }
                }
            }
    
    res = es_obj.search(index=index_name, body=query, size=n_results)
    return res

mecab = Mecab()
def morphs_split(text):
    global mecab
    text = mecab.morphs(text)
    return ' '.join(text)

def preprocess(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥<>()\s\.\?!》《≪≫\'<>〈〉:‘’%,『』「」＜＞・\"-“”∧]", "", text)
    return text

def context_split(text):
    text = ' '.join(text.strip().split('\\n')).strip()
    sent_list = text.strip().split('. ')
    text = ''
    for sent in sent_list:
        sent = preprocess(sent)
        sent = mecab.morphs(sent)
        text += ' '.join(sent)+'[SEP]'
    return text[:-5]

def sentence_split(text):
    text_list = [sent for sent in map(lambda x : x.strip(), text.split('[SEP]')) if sent != '']
    return text_list

def run_preprocess(data_dict):
    context = data_dict["text"]
    new_context = context_split(context)
    new_context =  ' '.join(sentence_split(new_context))
    data_dict["text"] = new_context
    return data_dict

def mk_new_file(mode, files, top_k, es):
    if mode == 'test':
        new_files = {'id':[], 'question':[], 'top_k':[]}
        for file in files:
            question_text = file['question']
            new_question = morphs_split(question_text)

            res = search_es(es_obj=es, index_name='nori-index', question_text=question_text, n_results=10)
            top_list = [hit['_source']['document_text'] for hit in res['hits']['hits']]

            new_files['id'].append(file['id'])
            new_files['question'].append(new_question)
            new_files['top_k'].append(top_list)
        return new_files
    
    else:
        new_files = {'context':[], 'id':[], 'question':[], 'top_k':[], 'answer_idx':[], 'answer':[], 'start_idx':[]}
        for file in files:
            context = file["context"]
            new_context = context_split(context)
            new_context =  ' '.join(sentence_split(new_context))

            question_text = file['question']
            res = search_es(es_obj=es, index_name='nori-index', question_text=question_text, n_results=top_k)
            top_list = [hit['_source']['document_text'] for hit in res['hits']['hits']]

            if not new_context in top_list:
                top_list = top_list[:-1] + [new_context]
                answer_idx = top_k-1
            else:
                answer_idx = top_list.index(new_context)

            answer = file['answers']['text'][0]
            new_answer = morphs_split(answer)

            start_idx = file["answers"]["answer_start"][0] 
            front_context = file["context"][:start_idx]
            back_context = file["context"][start_idx+len(answer):]

            new_front_context = context_split(front_context)
            new_back_context = context_split(back_context)
            new_front_context =  ' '.join(sentence_split(new_front_context))
            new_back_context = ' '.join(sentence_split(new_back_context))
            new_context = ' '.join([new_front_context, new_answer, new_back_context])

            ids_move = len(front_context) - len(new_front_context)
            start_idx = start_idx - ids_move + 1

            new_question = morphs_split(question_text)

            new_files['context'].append(new_context)
            new_files['id'].append(file['id'])
            new_files['question'].append(new_question)
            new_files['top_k'].append(top_list)
            new_files['answer_idx'].append(answer_idx)
            new_files['answer'].append(new_answer)
            new_files['start_idx'].append(start_idx)
        return new_files

def save_pickle(save_path, data_set):
    file = open(save_path, "wb")
    pickle.dump(data_set, file)
    file.close()

def get_pickle(pickle_path):
    f = open(pickle_path, "rb")
    dataset = pickle.load(f)
    f.close()
    return dataset

def mk_es():
    es_server = Popen(['elasticsearch-7.6.2/bin/elasticsearch'],
                   stdout=PIPE, stderr=STDOUT,
                   preexec_fn=lambda: os.setuid(1)  # as daemon
                  )
    # wait until ES has started
    print('wait...', end='\r')
    time.sleep(30)
    print('elastic sever connection!!')
    
    print('wait...', end='\r')
    config = {'host':'localhost', 'port':9200}
    es = Elasticsearch([config])

    # test connection
    es.ping()
    es.indices.delete(index='nori-index', ignore=[400, 404])
    index_config = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "nori_analyzer": {
                            "type": "custom",
                            "tokenizer": "nori_tokenizer",
                            "decompound_mode": "mixed",
                            "stopwords": "_korean_",
                        }
                    }
                }
            },
            "mappings": {
                "dynamic": "strict", 
                "properties": {
                    "document_text": {"type": "text", "analyzer": "nori_analyzer"}
                    }
                }
            }

    index_name = 'nori-index'
    es.indices.create(index=index_name, body=index_config, ignore=400)
    return es

def main(args):
    train_file = load_from_disk("/opt/ml/input/data/data/train_dataset")["train"]
    validation_file = load_from_disk("/opt/ml/input/data/data/train_dataset")["validation"]
    test_file = load_from_disk("/opt/ml/input/data/data/test_dataset")["validation"]
    
    wiki_path = os.path.join(args.save_path, 'dense_preprocess_wiki.json')
    if not os.path.isfile(wiki_path):
        print('No exist "dense_preprocess_wiki.json" !!')
        with open("/opt/ml/input/data/data/dense_preprocess_wiki.json", "r") as f:
            wiki = json.load(f)
        
        new_wiki = dict()
        iterator = trange(len(wiki), desc='wiki preprocess')
        for ids in iterator:
            new_wiki[str(ids)] = run_preprocess(wiki[str(ids)])
        
        with open(wiki_path, 'w', encoding='utf-8') as make_file:
            json.dump(new_wiki, make_file, indent="\t", ensure_ascii=False)
    
    with open(wiki_path, "r") as f:
        wiki = json.load(f)
    wiki_contexts = list(dict.fromkeys([v['text'] for v in wiki.values()]))

    qa_records = [{"example_id" : train_file[i]["id"], "document_title" : train_file[i]["title"], "question_text" : train_file[i]["question"], "answer" : train_file[i]["answers"]} for i in range(len(train_file))]
    all_wiki_articles = [{"document_text" : wiki_contexts[i]} for i in range(len(wiki_contexts))]

    es = mk_es()
    populate_index(es_obj=es, index_name='nori-index', evidence_corpus=all_wiki_articles)

    print('wait...', end='\r')
    new_train_file =  mk_new_file('train', train_file, args.top_k, es)
    print('make train dataset!!')
    
    print('wait...', end='\r')
    new_valid_file =  mk_new_file('valid', validation_file, args.top_k, es)
    print('make validation dataset!!')
    
    print('wait...', end='\r')
    new_test_file =  mk_new_file('test', test_file, args.top_k, es)
    print('make test dataset!!')
    
    save_pickle(os.path.join(args.save_path, f'Top{args.top_k}_preprocess_train.pkl'), new_train_file)
    save_pickle(os.path.join(args.save_path, f'Top{args.top_k}_preprocess_valid.pkl'), new_valid_file)
    save_pickle(os.path.join(args.save_path, f'Top{args.top_k}_preprocess_test.pkl'), new_test_file)
    print('complete!!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--save_path', type=str, default='/opt/ml/input/retrieval_dataset')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    print(f'TOP K ::: {args.top_k}')
    print(f'SAVE PATH ::: {args.save_path}')
    main(args)
