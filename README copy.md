# MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization


## 💡 Overview

<div align=left>
<img src=assets/logo.png width=90% />
</div>

## 📦 Requirements
1. Clone this repository and navigate to MMedPO folder
```bash
git clone https://github.com/Kelvinz-89757/MMedPO.git
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



## 📅 Schedule

- [ ] Release the data (VQA and report generation tasks)

- [ ] Release the training code


## 🙏Acknowledgement
We use code from [LLaVA-Med](https://github.com/microsoft/LLaVA-Med), [RULE](https://github.com/richard-peng-xia/RULE), [MedKLIP](https://github.com/MediaBrain-SJTU/MedKLIP). We thank the authors for releasing their code.
