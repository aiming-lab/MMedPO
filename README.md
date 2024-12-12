# MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization


## 💡 Overview

<div align=left>
<img src=assets/logo.png width=90% />
</div>

## 📦 Requirements
1. Clone this repository and navigate to MMedPO folder
```bash
git clone https://github.com/aiming-lab/MMedPO.git
cd MMedPO
```

2. Install Package: Create conda environment

```Shell
conda create -n MMedPO python=3.10 -y
conda activate MMedPO
pip install --upgrade pip  # enable PEP 660 support
pip install -r requirements.txt
pip install trl
```

3. Download the required model checkpoints [LLaVA-Med-1.5](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) from huggingface.

4. For all the medical datasets, you need firstly apply for the right of access and then download the dataset.

- [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
- [IU-Xray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view) (Thanks to [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT) for sharing the file)
- [VQA-RAD](https://osf.io/89kps/)
- [SLAKE](https://www.med-vqa.com/slake/)

## 🪧 Data Curation
We use MedKLIP to generate visual preference data. Use the following command or the script `inference_attention-map_score.sh` at `./scripts`

```Shell
python ./inference_attention-map_score.py \
    --config ./MedKLIP_config.yaml \
    --model_path /path/to/MedKLIP_model.pth \
    --dataset_name /dataset/name \
    --dataset_type caption \
    --image_root /path/to/dataset/image_folder \
    --annotation_save_root /path/to/save/annotation \
    --noised_image_save_root /path/to/save/noised_image \
```
## Train
Use the script `train_dpo_visual-text.sh` in `./scripts` or the following command, make sure to specify the necessary data paths and the checkpoint saving location.
```
deepspeed --include localhost:0,1,2,3 ./train/dpo/train_dpo_visual-text.py \
    --model_name_or_path /path/to/llava-med_model_checkpoint \
    --deepspeed ./scripts/zero3.json \
    --version v1 \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --data_path /path/to/data_json \
    --image_folder /path/to/img_folder \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /path/to/output_checkpoint_saving_location \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
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
```

## 🙏Acknowledgement
We use code from [LLaVA-Med](https://github.com/microsoft/LLaVA-Med), [RULE](https://github.com/richard-peng-xia/RULE), [MedKLIP](https://github.com/MediaBrain-SJTU/MedKLIP). We thank the authors for releasing their code.
