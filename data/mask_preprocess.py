import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def read_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]

    return lines


def encode_mask(mask, split=None, threshold=50):
    label_transformed = np.zeros(shape=mask.shape, dtype=np.uint8)

    if split == "red":
        red_mask = mask[:, :, 0] >= threshold
        label_transformed[red_mask] = [255, 0, 0] ## 1

    elif split == "green":
        green_mask = mask[:, :, 1] >= threshold
        label_transformed[green_mask] = [0, 255, 0] ## 2

    elif split == "rng":
        red_mask = mask[:, :, 0] >= threshold
        label_transformed[red_mask] = [255, 0, 0] ## 1
        green_mask = mask[:, :, 1] >= threshold
        label_transformed[green_mask] = [0, 255, 0] ## 2

    return label_transformed


def make_train_mask(total):
    make_dir(output_dir)
    for tmp in total:
        split, files = tmp[0], tmp[1]

        for idx in tqdm(range(len(files))):
            file = files[idx]
            mask = cv2.imread(f"{data_dir}/train_gt/{file}.jpeg")
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            encoded_mask = encode_mask(mask, split, threshold=100)

            encoded_mask = cv2.cvtColor(encoded_mask, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{output_dir}/{file}.jpeg", encoded_mask)


if __name__ == "__main__":
    data_dir = "/home/pervinco/Datasets/BKAI_IGH_NeoPolyp"
    output_dir = f"{data_dir}/train_mask"

    red_txt = f"{data_dir}/red.txt"
    green_txt = f"{data_dir}/green.txt"
    rng_txt = f"{data_dir}/rng.txt"

    red_files = read_txt(red_txt)
    green_files = read_txt(green_txt)
    rng_files = read_txt(rng_txt)

    print(len(red_files), len(green_files), len(rng_files))
    total_files = [["red", red_files], ["green",  green_files], ["rng", rng_files]]
    make_train_mask(total_files)