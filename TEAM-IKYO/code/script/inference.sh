python3 inference.py --output_dir ../output/baseline_pretrain/submission \
                    --model_name_or_path ../output/baseline_pretrain/baseline_pretrain.pt \
                    --tokenizer_name deepset/xlm-roberta-large-squad2 \
                    --config_name deepset/xlm-roberta-large-squad2 \
                    --retrieval_type elastic \
                    --retrieval_elastic_index wiki-index-split-800 \
                    --retrieval_elastic_num 35 \
                    --use_custom_model QAConvModelV2 \
                    --do_predict