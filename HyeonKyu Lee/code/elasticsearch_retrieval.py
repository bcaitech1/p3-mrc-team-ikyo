import json
import os
import time

from elasticsearch import Elasticsearch
from datasets import load_from_disk
from torch.utils.data import DataLoader, TensorDataset
from subprocess import Popen, PIPE, STDOUT
from tqdm import tqdm

# 엘라스틱 서치 노트북 파일 (es_retrieval.ipynb 를 먼저 실행하여 index 등록후 사용해야합니다. )
def elastic_setting():
    config = {'host':'localhost', 'port':9200}
    es = Elasticsearch([config])

    # 인덱스 이름
    index_name = "split-wiki-800-index" # ['split-wiki-index', 'nori-index', 'split-wiki-800-index']
    
    return es, index_name


def search_es(es_obj, index_name, question_text, n_results):
    # search query
    query = {
            'query': {
                'match': {
                    'document_text': question_text
                    }
                }
            }
    # n_result => 상위 몇개를 선택?
    res = es_obj.search(index=index_name, body=query, size=n_results)
    
    return res


def elastic_retrieval(es, index_name, question_text, n_results):
    res = search_es(es, index_name, question_text, n_results)
    # 매칭된 context만 list형태로 만든다.
    context_list = list((hit['_source']['document_text'], hit['_score']) for hit in res['hits']['hits'])
    return context_list

if __name__ == "__main__":
    es, index_name = elastic_setting()
    question_text = "대한민국의 대통령은?"
    context_list = elastic_retrieval(es, index_name, question_text, n_result)
    print(context_list)