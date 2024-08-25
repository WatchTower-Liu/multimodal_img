# CUDA_VISIBLE_DEVICES=2 python train.py --Textlora_rank 128 --Unetlora_rank 128 --per_device_train_batch_size 8 --gradient_accumulation_steps 1 --num_train_epochs 2 --save_steps 3000 --save_total_limit 5 --learning_rate 9e-5 --seed 42 --Textlora_dropout 0.10 --ddp_find_unused_parameters=False --feature_proj_lr 2e-4 --remove_unused_columns false --logging_steps 100 --output_dir ../../weights/train_V1_8 --Texttarget_modules "wte|c_attn|w1|w2|lm_head" --Unettarget_modules "to_k|to_q|to_v|to_out.0"

CUDA_VISIBLE_DEVICES=2 python train.py \
--Textlora_rank 128 \
--Unetlora_rank 128 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_train_epochs 4 \
--save_steps 3000 \
--save_total_limit 5 \
--learning_rate 9e-5 \
--seed 42 \
--Textlora_dropout 0.10 \
--ddp_find_unused_parameters false \
--feature_proj_lr 2e-4 \
--remove_unused_columns false \
--logging_steps 100 \
--output_dir ../../weights/train_V1_10 \
--Texttarget_modules "c_attn|w1|w2|lm_head"