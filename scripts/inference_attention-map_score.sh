cd ./curation/Sample_Zero-Shot_Grounding_RSNA || exit

# --data_type: caption, vqa
# --dataset_name: mimic, slake, vqa-rad, iuxray
python ./inference_attention-map_score.py \
    --config ./MedKLIP_config.yaml \
    --model_path /path/to/MedKLIP_model.pth \
    --dataset_name /dataset/name \
    --dataset_type caption \
    --image_root /path/to/dataset/image_folder \
    --annotation_save_root /path/to/save/annotation \
    --noised_image_save_root /path/to/save/noised_image \


