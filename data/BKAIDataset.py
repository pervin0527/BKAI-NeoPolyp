import cv2
import random
import numpy as np
import albumentations as A

from torch.utils.data import Dataset
from data.batch_preprocess import *

class BKAIDataset(Dataset):
    def __init__(self, config, split):
        super().__init__()
        self.base_dir = config["data_dir"]
        self.split = split
        self.size = config["img_size"]
        self.mean, self.std = config["mean"], config["std"]

        self.train_transform = A.Compose([A.Rotate(limit=90, p=0.6),
                                          A.HorizontalFlip(p=0.7),
                                          A.VerticalFlip(p=0.7),
                                          A.ColorJitter(brightness=(0.6,1.6), contrast=0.2, saturation=0.1, hue=0.01, p=0.5),
                                          A.Affine(scale=(0.5,1.5), translate_percent=(-0.125,0.125), rotate=(-180,180), shear=(-22.5,22), p=0.3),
                                          A.CoarseDropout(max_holes=1, max_height=100, max_width=100, p=0.4),
                                          A.ShiftScaleRotate(shift_limit_x=(-0.06, 0.06), shift_limit_y=(-0.06, 0.06), scale_limit=(-0.3, 0.1), rotate_limit=(-90, 90), border_mode=0, value=(0, 0, 0), p=0.8)])   
        
        self.image_transform = A.Compose([A.Blur(p=0.4),
                                          A.RandomBrightnessContrast(p=0.8),
                                          A.CLAHE(p=0.5)])

        self.split_txt = f"{self.base_dir}/{split}.txt"
        with open(self.split_txt, "r") as f:
            file_list = f.readlines()

        self.file_list = [x.strip() for x in file_list]   


    def __len__(self):
        return len(self.file_list)
        

    def __getitem__(self, index):
        file_name = self.file_list[index]
        image_file_path = f"{self.base_dir}/train/{file_name}.jpeg"
        mask_file_path = f"{self.base_dir}/train_gt/{file_name}.jpeg"
        image, mask = load_img_mask(image_file_path, mask_file_path, self.size)

        if self.split == "train":
            prob = random.random()
            if prob < 0.3:
                transform_image, transform_mask = train_img_mask_transform(self.train_transform, image, mask)

            elif 0.3 < prob < 0.6:
                piecies = [[image, mask]]
                while len(piecies) < 4:
                    idx = random.randint(0, len(self.file_list)-1)
                    file_name = self.file_list[idx]
                    piece_image = f"{self.base_dir}/train/{file_name}.jpeg"
                    piece_mask = f"{self.base_dir}/train_gt/{file_name}.jpeg"

                    piece_image, piece_mask = load_img_mask(piece_image, piece_mask, self.size)
                    transform_image, transform_mask = train_img_mask_transform(self.train_transform, piece_image, piece_mask)
                    piecies.append([transform_image, transform_mask])

                transform_image, transform_mask = mosaic_augmentation(piecies, self.size)

            else:
                idx = random.randint(0, len(self.file_list)-1)
                file_name = self.file_list[idx]
                piece_image = f"{self.base_dir}/train/{file_name}.jpeg"
                piece_mask = f"{self.base_dir}/train_gt/{file_name}.jpeg"

                piece_image, piece_mask = load_img_mask(piece_image, piece_mask, self.size)

                transform_image, transform_mask = cutmix_augmentation(image, mask, piece_image, piece_mask)

            if random.random() > 0.5:
                transform_image = train_image_transform(self.image_transform, image)
        
        else:
            transform_image = image
            transform_mask = mask
        
        transform_image = normalize(transform_image, self.mean, self.std)
        transform_mask = encode_mask(transform_mask)

        return transform_image, transform_mask