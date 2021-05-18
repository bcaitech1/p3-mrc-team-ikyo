python3 inference.py --output_dir /opt/ml/output/QAConvModel_afterpt_dr03_withQT/submission_r35_split  \
                    --model_name_or_path /opt/ml/output/QAConvModel_afterpt_dr03_withQT/QAConvModel_afterpt_dr03_withQT.pt \
                    --tokenizer_name deepset/xlm-roberta-large-squad2 \
                    --config_name deepset/xlm-roberta-large-squad2 \
                    --retrival_type elastic \
                    --use_custom_model \
                    --do_predict