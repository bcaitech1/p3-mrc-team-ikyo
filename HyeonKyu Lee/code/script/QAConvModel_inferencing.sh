python3 inference.py --output_dir /opt/ml/output/hk_V2_QAConvModel_dr03/submission_r35_split800 \
                    --model_name_or_path /opt/ml/output/hk_V2_QAConvModel_dr03/hk_V2_QAConvModel_dr03.pt \
                    --tokenizer_name deepset/xlm-roberta-large-squad2 \
                    --config_name deepset/xlm-roberta-large-squad2 \
                    --retrival_type elastic \
                    --use_custom_model \
                    --do_predict