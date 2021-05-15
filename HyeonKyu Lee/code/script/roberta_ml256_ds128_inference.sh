python3 inference.py --output_dir /opt/ml/output/hk_ml256_ds128/submission  \
                    --model_name_or_path /opt/ml/output/hk_ml256_ds128/hk_ml256_ds128.pt \
                    --tokenizer_name xlm-roberta-large \
                    --config_name xlm-roberta-large \
                    --retrival_type elastic \
                    --do_predict