import os
import cv2
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from data.BKAIDataset import BKAIDataset


def predict(epoch, config, img_size, model, device):
    model.eval()

    dataset = BKAIDataset(config["data_dir"], split="valid", size=config["img_size"])
    data_dir = config["data_dir"]
    save_dir = config["save_dir"]
    num_samples = config["num_pred_samples"]

    if not os.path.isdir(f"{save_dir}/predict"):
        os.makedirs(f"{save_dir}/predict")

    with open(f"{data_dir}/valid.txt", "r") as f:
        files = f.readlines()

    random.shuffle(files)
    samples = random.sample(files, num_samples)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 25))
    for idx, sample in enumerate(samples):
        file = sample.strip()

        image = cv2.imread(f"{data_dir}/train/{file}.jpeg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(f"{data_dir}/train_gt/{file}.jpeg")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        overlayed = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

        x = cv2.resize(image, (img_size, img_size))
        x = dataset.normalize(x)
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x).to(device)

        y_pred = model(x)
        pred_mask = torch.argmax(y_pred[0], dim=0).cpu().numpy()

        color_decoded = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        color_decoded[pred_mask == 0] = [0, 0, 0]
        color_decoded[pred_mask == 1] = [255, 0, 0]
        color_decoded[pred_mask == 2] = [0, 255, 0]
        color_decoded = cv2.resize(color_decoded, (mask.shape[1], mask.shape[0]))

        axes[idx, 0].imshow(overlayed)
        axes[idx, 0].set_title("Original Mask")

        axes[idx, 1].imshow(color_decoded)
        axes[idx, 1].set_title("Predict Mask")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/predict/epoch_{epoch:>04}.png")
    plt.close()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")


def save_config_to_yaml(config, save_dir):
    with open(f"{save_dir}/params.yaml", 'w') as file:
        yaml.dump(config, file)


def encode_mask(mask, threshold):
    label_transformed = np.full(mask.shape[:2], 0, dtype=np.uint8)

    red_mask = (mask[:, :, 0] > threshold) & (mask[:, :, 1] < 50) & (mask[:, :, 2] < 50)
    label_transformed[red_mask] = 1

    green_mask = (mask[:, :, 0] < 50) & (mask[:, :, 1] > threshold) & (mask[:, :, 2] < 50)
    label_transformed[green_mask] = 2

    return label_transformed


def calculate_effective_samples(config):
    beta = 0.9999
    class_pixels = [0] * config["num_classes"]
    mask_dir = config["data_dir"] + "/train_gt"
    mask_files = sorted(glob(f"{mask_dir}/*.jpeg"))

    for file in mask_files:
        mask = cv2.imread(file)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        encoded_mask = encode_mask(mask, config["mask_threshold"])

        for i in range(3):
            class_pixels[i] += (encoded_mask == i).sum().item()

    effective_samples = [(1.0 - beta) / (1.0 - beta**count) for count in class_pixels]

    return torch.tensor(effective_samples, dtype=torch.float32)