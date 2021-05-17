python3 inference.py --output_dir /opt/ml/output/baseline-roberta/submission_retrival30 \
                    --model_name_or_path /opt/ml/output/baseline-roberta/baseline-roberta.pt \
                    --tokenizer_name xlm-roberta-large \
                    --config_name xlm-roberta-large \
                    --retrival_type elastic \
                    --do_predict