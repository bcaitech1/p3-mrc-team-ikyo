import os
import json
import time
from tqdm import tqdm

from elasticsearch import Elasticsearch
from datasets import load_from_disk
from torch.utils.data import DataLoader, TensorDataset
from subprocess import Popen, PIPE, STDOUT

from prepare_dataset import make_custom_dataset


def populate_index(es_obj, index_name, evidence_corpus):

    for i, rec in enumerate(tqdm(evidence_corpus)):
        try:
            index_status = es_obj.index(index=index_name, id=i, body=rec)
        except:
            print(f'Unable to load document {i}.')
            
    n_records = es_obj.count(index=index_name)['count']
    print(f'Succesfully loaded {n_records} into {index_name}')
    return


def set_datas() :
    if not os.path.isfile("/opt/ml/input/data/preprocess_train.pkl") :
        make_custom_dataset("/opt/ml/input/data/preprocess_train.pkl")
    train_file = load_from_disk("/opt/ml/input/data/data/train_dataset")["train"]
    validation_file = load_from_disk("/opt/ml/input/data/data/train_dataset")["validation"]

    with open("/opt/ml/outer_datas/split_wiki.json", "r") as f:
        wiki = json.load(f)
    wiki_contexts = list(dict.fromkeys([v['text'] for v in wiki.values()]))

    qa_records = [{"example_id" : train_file[i]["id"], "document_title" : train_file[i]["title"], "question_text" : train_file[i]["question"], "answer" : train_file[i]["answers"]} for i in range(len(train_file))]
    wiki_articles = [{"document_text" : wiki_contexts[i]} for i in range(len(wiki_contexts))]
    return qa_records, wiki_articles


def set_index_and_server() :
    es_server = Popen(['/opt/ml/elasticsearch-7.6.2/bin/elasticsearch'],
                    stdout=PIPE, stderr=STDOUT,
                    preexec_fn=lambda: os.setuid(1)  # as daemon
                    )
    time.sleep(30)

    config = {'host':'localhost', 'port':9200}
    es = Elasticsearch([config])

    index_config = {
        "settings": {
            "analysis": {
                "filter":{
                    "my_stop_filter": {
                        "type" : "stop",
                        "stopwords_path" : "user_dic/my_stop_dic.txt"
                    }
                },
                "analyzer": {
                    "nori_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "decompound_mode": "mixed",
                        "filter" : ["my_stop_filter"]
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

    index_name = 'split-wiki-index'
    print('elastic serach ping :', es.ping())
    print(es.indices.create(index=index_name, body=index_config, ignore=400))

    return es, index_name


def main() :
    print('Start to Set Elastic Search')
    print('It takes almost 8 minutes')
    _, wiki_articles = set_datas()
    es, index_name = set_index_and_server()
    populate_index(es_obj=es, index_name=index_name, evidence_corpus=wiki_articles)
    print('Finish')


if __name__ == '__main__' : 
    main()