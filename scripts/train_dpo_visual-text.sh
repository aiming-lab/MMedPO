export CUDA_VISIBLE_DEVICES='0,1,2,3'
CUDA="0,1,2,3"

cd ./train/dpo || exit
deepspeed --include localhost:$CUDA --master_port $((RANDOM + 30000)) ./train_dpo_weighted.py \
    --model_name_or_path /path/to/model/checkpoint \
    --deepspeed ./scripts/zero3.json \
    --version v1 \
    --loss_use_weight True \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --data_path /path/to/train_json_file \
    --image_folder /path/to/image_folder  \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /path/to/output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to wandb \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \




