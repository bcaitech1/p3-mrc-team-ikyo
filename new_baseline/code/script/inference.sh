python3 inference.py --output_dir /opt/ml/submission  \
                    --model_name_or_path /opt/ml/mycode/code/output/xlm-roberta-large.pt \
                    --tokenizer_name xlm-roberta-large \
                    --config_name xlm-roberta-large \
                    --retrival_type elastic \
                    --use_custom_model \
                    --do_predict