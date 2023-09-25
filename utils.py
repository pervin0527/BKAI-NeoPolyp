import os
import cv2
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from data.batch_preprocess import *

def decode_image(image):
    image = np.transpose(image, (1, 2, 0))
    image = image * 255
    image = image.astype(np.uint8)

    return image


def decode_mask(pred_mask):
        decoded_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        decoded_mask[pred_mask == 0] = [0, 0, 0]
        decoded_mask[pred_mask == 1] = [0, 255, 0] ## Green
        decoded_mask[pred_mask == 2] = [255, 0, 0] ## Red
        
        return decoded_mask


def predict(epoch, config, model, device):
    model.eval()

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
        img_path = f"{data_dir}/train/{file}.jpeg"
        mask_path = f"{data_dir}/train_gt/{file}.jpeg"

        image, mask = load_img_mask(img_path, mask_path, size=config["img_size"])
        x = normalize(image, config["mean"], config["std"])
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x).to(device)

        y_pred = model(x)
        pred_mask = torch.argmax(y_pred[0], dim=0).cpu().numpy()
        
        decoded_mask = decode_mask(pred_mask)
        decoded_mask = cv2.resize(decoded_mask, (mask.shape[1], mask.shape[0]))

        overlayed = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
        axes[idx, 0].imshow(overlayed)
        axes[idx, 0].set_title("Original Mask")

        axes[idx, 1].imshow(decoded_mask)
        axes[idx, 1].set_title("Predict Mask")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/predict/epoch_{epoch:>04}.png")
    plt.close()


def make_test_txt(dir, split_name):
    files = sorted(glob(f"{dir}/{split_name}/*.jpeg"))
    with open(f"{dir}/test.txt", "w") as f:
        for idx, file in enumerate(files):
            name = file.split('/')[-1].split('.')
            f.write(name)

            if idx != len(files):
                f.write("\n")


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


def visualize(images, masks):
    assert len(images) == len(masks), "Length of images and masks should be the same."

    num_rows = len(images)
    plt.figure(figsize=(10, 4 * num_rows))
    
    for idx, (image, mask) in enumerate(zip(images, masks)):
        # Image
        plt.subplot(num_rows, 2, 2 * idx + 1)
        plt.imshow(image)
        plt.title(f"Image {idx + 1}")
        
        # Mask
        plt.subplot(num_rows, 2, 2 * idx + 2)
        plt.imshow(mask)
        plt.title(f"Mask {idx + 1}")
