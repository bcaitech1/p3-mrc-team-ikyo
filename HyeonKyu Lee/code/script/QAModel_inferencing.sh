python3 inference.py --output_dir /opt/ml/output/hk_QAModel/submission  \
                    --model_name_or_path /opt/ml/output/hk_QAModel/hk_QAModel.pt \
                    --tokenizer_name xlm-roberta-large \
                    --config_name xlm-roberta-large \
                    --retrival_type elastic \
                    --use_custom_model \
                    --do_predict