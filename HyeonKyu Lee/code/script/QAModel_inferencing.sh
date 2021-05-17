python3 inference.py --output_dir /opt/ml/output/hk_QAModel_after_ptdr01_trdr07/submission_r30_split  \
                    --model_name_or_path /opt/ml/output/hk_QAModel_after_ptdr01_trdr07/hk_QAModel_after_ptdr01_trdr07.pt \
                    --tokenizer_name xlm-roberta-large \
                    --config_name xlm-roberta-large \
                    --retrival_type elastic \
                    --use_custom_model \
                    --do_predict