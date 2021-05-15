python3 train_mrc.py --output_dir /opt/ml/output \
                    --model_name_or_path /opt/ml/output/sun_mrc_pretrain/koquard_pretrained_model.pt \
                    --tokenizer_name xlm-roberta-large \
                    --config_name xlm-roberta-large \
                    --learning_rate 1e-6 \
                    --num_train_epoch 3 \
                    --per_device_train_batch_size 16 \
                    --per_device_eval_batch_size 16 \
                    --dataset_name concat \
                    --use_pretrained_koquard_model \
                    --run_name sun_mrc_train_pretrained_epoch_1_mask_augmentation
                    # --use_custom_model \
