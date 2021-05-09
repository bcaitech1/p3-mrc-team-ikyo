# README.md

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
    \_output # Directory 생성 필요
```

## Directory Structure

```bash
\_opt\_ml\
    \_input\_data
				\_data # 주최측 제공 Data
						\_ ...
        \_add_squad_kor_v1_2.pkl
				\_preprocess_train.pkl
				\_train_concat5.pkl
				\_preprocess_wiki.json
    \_output
				\_ korquard_pretrained_model
						\_korquard_pretrained_model.pt
				\_run_name # 매 실험마다 생성
						\_run_name.pt # 매 실험마다 생성
						\_run_name.pt # 매 실험마다 생성
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

## Pretrain korQuAD(optional)

```bash
python3 train_mrc.py --output_dir /opt/ml/output \
                        --model_name_or_path xlm-roberta-large \
                        --tokenizer_name xlm-roberta-large \
                        --config_name xlm-roberta-large \
                        --learning_rate 0.00001 \
                        --num_train_epoch 5 \
                        --per_device_train_batch_size 16 \
                        --per_device_eval_batch_size 16 \
                        --dataset_name only_korquad \
                        --use_custom_model \  # custom_model 사용하지 않을 경우 주석 처리
                        --run_name korquard_pretrained_model
```

## Train mrc

```bash
python3 train_mrc.py --output_dir /opt/ml/output \
                        --model_name_or_path /opt/ml/output/korquard_pretrained_model/korquard_pretrained_model.pt \ #(inference 하고자하는 모델의 경로)
                        --tokenizer_name xlm-roberta-large \
                        --config_name xlm-roberta-large \
                        --use_pretrained_koquard_model \
                        --use_custom_model \
                        --learning_rate 0.00001 \
                        --num_train_epoch 5 \
                        --per_device_train_batch_size 16 \
                        --per_device_eval_batch_size 16 \
	                      --dataset_name concat \ # 'basic', 'preprocess', 'concat', 'korquad', 'only_korquad'
                        --run_name ***run_name*** # run_name은 바꿔야 함.
```

## Inference

```bash
# Inference
python3 inference.py --output_dir /opt/ml/output/***run_name*** \ # run_name은 바꿔야 함. 
                    --model_name_or_path /opt/ml/output/***run_name***/***run_name***.pt \ # run_name은 바꿔야 함. (inference 하고자하는 모델의 경로)
                    --tokenizer_name xlm-roberta-large \
                    --config_name xlm-roberta-large \
                    --retrival_type elastic \
                    --use_custom_model # custom_model 사용하지 않을 경우 주석 처리
```