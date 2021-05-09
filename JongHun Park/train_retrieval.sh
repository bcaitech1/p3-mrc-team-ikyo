python3 train_retrieval.py --output_dir /opt/ml/retrieval_output/ \
                           --model_checkpoint bert-base-multilingual-cased \
                           --seed 2021 \
                           --epoch 10 \
                           --learning_rate 1e-5 \
                           --gradient_accumulation_steps 1 \
                           --top_k 20 \
                           --run_name dense_retrieval
