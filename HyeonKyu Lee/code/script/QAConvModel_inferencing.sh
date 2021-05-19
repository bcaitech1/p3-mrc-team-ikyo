python3 inference.py --output_dir /opt/ml/output/hk_QAConvModel_fixed_deepset_dr07_withQT_NoTruncVV_T10/submission_r35_split \
                    --model_name_or_path /opt/ml/output/hk_QAConvModel_fixed_deepset_dr07_withQT_NoTruncVV_T10/hk_QAConvModel_fixed_deepset_dr07_withQT_NoTruncVV_T10.pt \
                    --tokenizer_name deepset/xlm-roberta-large-squad2 \
                    --config_name deepset/xlm-roberta-large-squad2 \
                    --retrival_type elastic \
                    --use_custom_model \
                    --do_predict