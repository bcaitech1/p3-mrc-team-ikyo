python3 train_mrc.py --output_dir /opt/ml/output \
                        --model_name_or_path /opt/ml/output/hk_QAModel_pretraining/koquard_pretrained_model.pt \
                        --tokenizer_name xlm-roberta-large \
                        --config_name xlm-roberta-large \
                        --use_pretrained_koquard_model \
                        --use_custom_model \
                        --learning_rate 0.00001 \
                        --num_train_epoch 5 \
                        --per_device_train_batch_size 16 \
                        --per_device_eval_batch_size 16 \
                        --dataset_name concat \
                        --run_name hk_QAModel_after_pt








