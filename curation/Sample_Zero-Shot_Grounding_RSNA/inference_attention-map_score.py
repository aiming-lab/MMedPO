import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import sys

from dataset.dataset_RSNA import RSNA2018_Dataset
from models.model_MedKLIP import MedKLIP
from models.tokenization_bert import BertTokenizer
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from PIL import Image

original_class = [
    "normal",
    "clear",
    "sharp",
    "sharply",
    "unremarkable",
    "intact",
    "stable",
    "free",
    "effusion",
    "opacity",
    "pneumothorax",
    "edema",
    "atelectasis",
    "tube",
    "consolidation",
    "process",
    "abnormality",
    "enlarge",
    "tip",
    "low",
    "pneumonia",
    "line",
    "congestion",
    "catheter",
    "cardiomegaly",
    "fracture",
    "air",
    "tortuous",
    "lead",
    "disease",
    "calcification",
    "prominence",
    "device",
    "engorgement",
    "picc",
    "clip",
    "elevation",
    "expand",
    "nodule",
    "wire",
    "fluid",
    "degenerative",
    "pacemaker",
    "thicken",
    "marking",
    "scar",
    "hyperinflate",
    "blunt",
    "loss",
    "widen",
    "collapse",
    "density",
    "emphysema",
    "aerate",
    "mass",
    "crowd",
    "infiltrate",
    "obscure",
    "deformity",
    "hernia",
    "drainage",
    "distention",
    "shift",
    "stent",
    "pressure",
    "lesion",
    "finding",
    "borderline",
    "hardware",
    "dilation",
    "chf",
    "redistribution",
    "aspiration",
    "tail_abnorm_obs",
    "excluded_obs",
]


def get_tokenizer(tokenizer, target_text):

    target_tokenizer = tokenizer(
        list(target_text),
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    return target_tokenizer


def score_cal(labels, seg_map, pred_map):
    """
    labels B * 1
    seg_map B *H * W
    pred_map B * H * W
    """
    device = labels.device
    total_num = torch.sum(labels)
    mask = (labels == 1).squeeze()
    seg_map = seg_map[mask, :, :].reshape(total_num, -1)
    pred_map = pred_map[mask, :, :].reshape(total_num, -1)
    one_hot_map = pred_map > 0.008
    dot_product = (seg_map * one_hot_map).reshape(total_num, -1)

    max_number = torch.max(pred_map, dim=-1)[0]
    point_score = 0
    for i, number in enumerate(max_number):
        temp_pred = (pred_map[i] == number).type(torch.int)
        flag = int((torch.sum(temp_pred * seg_map[i])) > 0)
        point_score = point_score + flag
    mass_score = torch.sum(dot_product, dim=-1) / (
        (torch.sum(seg_map, dim=-1) + torch.sum(one_hot_map, dim=-1))
        - torch.sum(dot_product, dim=-1)
    )
    dice_score = (
        2
        * (torch.sum(dot_product, dim=-1))
        / (torch.sum(seg_map, dim=-1) + torch.sum(one_hot_map, dim=-1))
    )
    return total_num, point_score, mass_score.to(device), dice_score.to(device)


def visualize_heatmap_on_image(image_tensor, heatmap_tensor, save_path):
    image_tensor = image_tensor.squeeze(0)
    image_np = image_tensor.permute(1, 2, 0).numpy()
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

    heatmap_np = heatmap_tensor.squeeze(0).numpy()
    heatmap_np = (heatmap_np - np.min(heatmap_np)) / (
        np.max(heatmap_np) - np.min(heatmap_np)
    )

    plt.imshow(image_np)
    plt.imshow(heatmap_np, cmap="jet", alpha=0.5)
    plt.axis("off")

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


class DPODateset:
    def __init__(self, file_path, args):
        self.data = json.load(open(file_path, "r"))
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.seg_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([224, 224], interpolation=InterpolationMode.NEAREST),
            ]
        )
        self.image_root = args.image_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        idx = sample["id"]
        img = sample["image"]
        img_path = os.path.join(self.image_root, img)
        img = Image.open(img_path).convert("RGB")
        image = self.transform(img)
        seg_map = np.zeros((1024, 1024))
        seg_map = self.seg_transfrom(seg_map)
        question = sample["conversations"][0]["value"]
        correct_answer = sample["conversations"][1]["value"]

        # image = torch.tensor(image).unsqueeze(0).float()
        class_label = np.array([0])

        return {
            "id": idx,
            "image": image,
            "label": class_label,
            "image_path": img_path,
            "seg_map": seg_map,
            "question": question,
            "correct_answer": correct_answer,
        }


def generate_noised_image(image_paths, pred_maps):
    noised_images = []

    for i in range(len(image_paths)):
        # Load the original image
        original_image = Image.open(image_paths[i])
        original_image = np.array(original_image)

        # Resize pred_map to match the original image size using nearest interpolation
        pred_map = pred_maps[i]
        # pred_map_resized = cv2.resize(pred_map, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        pred_map_resized = cv2.resize(
            pred_map,
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        pred_map_resized = cv2.GaussianBlur(pred_map_resized, (15, 15), 0)
        mean_value = np.mean(pred_map_resized)
        std_value = np.std(pred_map_resized)

        max_noise_strength = 10
        # threshold = mean_value + std_value
        threshold = mean_value - std_value
        # percentile_value = np.percentile(pred_map_resized, 90)
        # Create a Gaussian noise mask based on the pred_map_resized
        noise = np.random.normal(
            0, 10, original_image.shape
        )  # Adjust the mean and variance as needed

        # pred_map_resized_expanded = np.expand_dims(pred_map_resized, axis=-1)  # Shape will become (624, 512, 1)
        # pred_map_resized_expanded = np.repeat(pred_map_resized_expanded, 3, axis=-1)  # Now shape is (624, 512, 3)

        # weighted_noise = noise * (pred_map_resized_expanded / np.max(pred_map_resized_expanded))* max_noise_strength

        noised_image = original_image.copy()
        # noised_image = noised_image.astype(np.float32)

        noised_image[pred_map_resized > threshold] = noise[pred_map_resized > threshold]
        # noised_image[pred_map_resized > threshold] += weighted_noise[pred_map_resized > threshold]

        noised_image = np.clip(noised_image, 0, 255).astype(np.uint8)

        # Append the noised image to the list
        noised_images.append(noised_image)

    return noised_images


def generate_weighted_score(scores, label_index):
    weighted_scores = []
    for i in range(len(scores)):
        binary_index_score = scores[i][label_index]
        softmax_score = nn.functional.softmax(binary_index_score, dim=-1)
        weighted_score = softmax_score[1]
        weighted_scores.append(weighted_score)
    return weighted_scores


def main(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Total CUDA devices: ", torch.cuda.device_count())
    torch.set_default_tensor_type("torch.FloatTensor")

    #### Dataset ####
    print("Creating dataset")
    # test_dataset =  RSNA2018_Dataset(config['test_file'])
    test_dataset = DPODateset(config["test_file"], args)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["test_batch_size"],
        num_workers=32,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=None,
        drop_last=False,
    )
    json_book = json.load(open(config["disease_book"], "r"))
    disease_book = [json_book[i] for i in json_book]
    ana_book = [
        "It is located at " + i
        for i in [
            "trachea",
            "left_hilar",
            "right_hilar",
            "hilar_unspec",
            "left_pleural",
            "right_pleural",
            "pleural_unspec",
            "heart_size",
            "heart_border",
            "left_diaphragm",
            "right_diaphragm",
            "diaphragm_unspec",
            "retrocardiac",
            "lower_left_lobe",
            "upper_left_lobe",
            "lower_right_lobe",
            "middle_right_lobe",
            "upper_right_lobe",
            "left_lower_lung",
            "left_mid_lung",
            "left_upper_lung",
            "left_apical_lung",
            "left_lung_unspec",
            "right_lower_lung",
            "right_mid_lung",
            "right_upper_lung",
            "right_apical_lung",
            "right_lung_unspec",
            "lung_apices",
            "lung_bases",
            "left_costophrenic",
            "right_costophrenic",
            "costophrenic_unspec",
            "cardiophrenic_sulcus",
            "mediastinal",
            "spine",
            "clavicle",
            "rib",
            "stomach",
            "right_atrium",
            "right_ventricle",
            "aorta",
            "svc",
            "interstitium",
            "parenchymal",
            "cavoatrial_junction",
            "cardiopulmonary",
            "pulmonary",
            "lung_volumes",
            "unspecified",
            "other",
        ]
    ]
    tokenizer = BertTokenizer.from_pretrained(config["text_encoder"])
    ana_book_tokenizer = get_tokenizer(tokenizer, ana_book).to(device)
    disease_book_tokenizer = get_tokenizer(tokenizer, disease_book).to(device)

    print("Creating model")
    model = MedKLIP(config, ana_book_tokenizer, disease_book_tokenizer, mode="train")
    model = nn.DataParallel(
        model, device_ids=[i for i in range(torch.cuda.device_count())]
    )
    model = model.to(device)

    checkpoint = torch.load(args.model_path, map_location="cpu")
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    print("load checkpoint from %s" % args.model_path)

    print("Start testing")
    model.eval()
    json_path = os.path.join(
        args.annotation_save_root,
        args.dataset_name + f"_visual_dpo-{args.dataset_type}.json",
    )
    # ans_file = open(json_path, "w")
    ans_items = []
    for i, sample in enumerate(tqdm(test_dataloader, position=0, file=sys.stdout)):
        ids = sample["id"]
        images = sample["image"].to(device)
        image_paths = sample["image_path"]
        batch_size = images.shape[0]
        labels = sample["label"].to(device)
        seg_map = sample["seg_map"][:, 0, :, :].to(device)  # B C H W
        questions = sample["question"]
        correct_answers = sample["correct_answer"]

        with torch.no_grad():
            scores, ws = model(
                images, labels, is_train=False
            )  # batch_size,batch_size,image_patch,text_patch
            ws = (ws[-4] + ws[-3] + ws[-2] + ws[-1]) / 4
            ws = ws.reshape(batch_size, ws.shape[1], 14, 14)

            label_index = original_class.index("disease")
            pred_maps = (
                ws[:, original_class.index("disease"), :, :].detach().cpu().numpy()
            )

            pred_maps = torch.from_numpy(
                pred_maps.repeat(16, axis=1).repeat(16, axis=2)
            ).to(
                device
            )  # Final
            images_cpu = images.cpu().numpy()
            pred_maps_cpu = pred_maps.cpu().numpy()
            
            # print(ids)
            ids = [str(i.item()) for i in ids]
            noised_image_paths = [
                os.path.join(
                    args.noised_image_save_root, args.dataset_name, ids[i] + ".png"
                )
                for i in range(batch_size)
            ]
            rejected_images = generate_noised_image(image_paths, pred_maps_cpu)

            for i in range(batch_size):
                cv2.imwrite(noised_image_paths[i], rejected_images[i])
            weighted_scores = generate_weighted_score(scores, label_index)
            for (
                idx,
                image_path,
                noised_image_path,
                question,
                correct_answer,
                weighted_score,
            ) in zip(
                ids,
                image_paths,
                noised_image_paths,
                questions,
                correct_answers,
                weighted_scores,
            ):
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                if isinstance(weighted_score, torch.Tensor):
                    weighted_score = weighted_score.item()
                dpo_item = {
                    "id": idx,
                    "image": image_path,
                    "rejected_image": noised_image_path,
                    "weighted_score": weighted_score,
                    "conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": correct_answer},
                    ],
                    "rejected_conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": correct_answer},
                    ],
                }
                # ans_file.write(json.dumps(dpo_item)+'\n')
                # ans_file.flush()
                ans_items.append(dpo_item)
    # ans_file.close()
    # with open(json_path, "w") as ans_file:
    #     json.dump(ans_items, ans_file, indent=4)
    # get the map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./MedKLIP_config.yaml")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=str, default="1", help="gpu")
    parser.add_argument("--noised_image_save_root", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="DPO")
    parser.add_argument("--annotation_save_root", type=str, default="")
    parser.add_argument("--dataset_type", type=str, default="")
    parser.add_argument("--image_root", type=str, default="")

    args = parser.parse_args()
    # yaml = YAML(typ='rt')
    # yaml.load(...)
    # config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    from ruamel.yaml import YAML

    yaml = YAML(typ="rt")  # 'rt' means round-trip
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu != "-1":
        torch.cuda.current_device()
        torch.cuda._initialized = True

    main(args, config)
