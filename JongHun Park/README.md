# Dense Retrieval

## Directory Setting

```bash
\_opt\_ml\
    \_input\_data
	\_ data # 주최측 제공 Data
		\_ dummy_dataset
			\_ ...
		\_ test_dataset
			\_ ...
		\_ train_dataset
			\_ ...
		\_ wikipedia_documents.json
```

## Directory Structure

```bash
python3 mk_retrieval_dataset.py --top_k 20 \
                                --save_path /opt/ml/input/retrieval_dataset
```

```bash
\_opt\_ml\
    \_input\_retrieval_dataset
		\_dense_preprocess_wiki.json
		\_Top***k***_preprocess_test.pkl  # k는 top_k로 설정됨
		\_Top***k***_preprocess_train.pkl
		\_Top***k***_preprocess_valid.pkl
    \_retrieval_output
	\_run_name # 매 실험마다 생성
            \_model
		\_p_run_name.pt
                \_q_run_name.pt
```

## Installation

```bash
# download elasticsearch
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.2-linux-x86_64.tar.gz -q
tar -xzf elasticsearch-7.6.2-linux-x86_64.tar.gz
chown -R daemon:daemon elasticsearch-7.6.2

# install elasticsearch library
pip install elasticsearch

# install nori tokenizer
elasticsearch-7.6.2/bin/elasticsearch-plugin install analysis-nori
```

## Train Retrieval

```bash
python3 train_retrieval.py --output_dir /opt/ml/retrieval_output/ \
                           --model_checkpoint bert-base-multilingual-cased \
                           --seed 2021 \
                           --epoch 10 \
                           --learning_rate 1e-5 \
                           --gradient_accumulation_steps 1 \
                           --top_k 20 \
                           --run_name dense_retrieval
```
