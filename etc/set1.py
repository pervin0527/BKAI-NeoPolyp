import os
import cv2
import random
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class BKAIDataset(Dataset):
    def __init__(self, base_dir, split="train", size=256, threshold=50, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.split = split
        self.size = (size, size)
        self.threshold = int(threshold)

        self.mean = mean
        self.std = std

        self.base_dir = base_dir
        self.image_dir = f"{base_dir}/train"
        self.mask_dir = f"{base_dir}/train_gt"

        with open(f"{self.base_dir}/{self.split}.txt", "r") as f:
            lines = f.read().splitlines()

        self.images, self.masks = [], []
        for line in lines:
            image = f"{self.image_dir}/{line}.jpeg"
            mask = f"{self.mask_dir}/{line}.jpeg"

            self.images.append(image)
            self.masks.append(mask)

        print(f"Number of images in {self.split} : {len(self.images)}")


    def __len__(self):
        return len(self.images)
    
    
    def __getitem__(self, index):
        image, mask = self.load_image_mask(self.images[index])

        if self.split == "train":
            if random.random() < 0.5:
                t_image, t_mask = self.train_transform(image, mask)
            else:
                samples = [(image, mask)]
                for _ in range(3):
                    i = random.randint(0, len(self.images)-1)
                    image_i, mask_i = self.load_image_mask(self.images[i])
                    samples.append((image_i, mask_i))
                
                t_image, t_mask = self.mosaic_augmentation(samples, (256, 256))
        else:
            t_image, t_mask = self.valid_transform(image, mask)

        # t_image = cv2.resize(image, (256, 256))
        # t_mask = cv2.resize(mask, (256, 256))

        # batch_x = self.tensor_transform(t_image)
        batch_x = np.transpose(t_image, (2, 0, 1))
        batch_x = batch_x / 255.0
        batch_y = self.encode_mask(t_mask)

        return batch_x, batch_y
    

    def load_image_mask(self, file):
        image = cv2.imread(file)
        mask = cv2.imread(file)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        return image, mask


    def encode_mask(self, mask):
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        red_mask = (mask[:, :, 0] > self.threshold) & (mask[:, :, 1] < 50) & (mask[:, :, 2] < 50)
        label_mask[red_mask] = 1

        green_mask = (mask[:, :, 0] < 50) & (mask[:, :, 1] > self.threshold) & (mask[:, :, 2] < 50)
        label_mask[green_mask] = 2

        return label_mask
    

    def train_transform(self, image, mask):
        transform = A.Compose([A.OneOf([A.Resize(256, 256, p=0.5),
                               A.RandomCrop(height=256, width=256, p=0.0)], p=1),
                           
                               A.OneOf([A.Flip(p=0.3),
                               A.ShiftScaleRotate(shift_limit_x=(-0.06, 0.06),
                                                   shift_limit_y=(-0.06, 0.06), 
                                                   scale_limit=(-0.3, 0.1),
                                                   rotate_limit=(-90, 90),
                                                   interpolation=0,
                                                   border_mode=0,
                                                   value=(0, 0, 0),
                                                   mask_value=None, 
                                                   rotate_method='largest_box',
                                                   p=0.7)], p=1),])
        
        transformed = transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        return transformed_image, transformed_mask
    

    def valid_transform(self, image, mask):
        transform = A.Compose([A.Resize(256, 256, p=1)])
        
        transformed = transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        return transformed_image, transformed_mask
    

    def tensor_transform(self, image):
        transform = transforms.Compose([transforms.ToTensor(),
                                        # transforms.Normalize(mean=self.mean, std=self.std),
                                        ])
        
        return transform(image)
    

    def mosaic_augmentation(self, samples, output_size=(256, 256)):
        h, w = output_size
        mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)
        mosaic_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 중심점 지정
        cx, cy = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
        
        indices = [0, 1, 2, 3]
        random.shuffle(indices)
        for i, index in enumerate(indices):
            image, mask = samples[index]
            if i == 0:
                mosaic_img[:cy, :cx] = cv2.resize(image, (cx, cy))
                mosaic_mask[:cy, :cx] = cv2.resize(mask, (cx, cy))
            elif i == 1:
                mosaic_img[:cy, cx:] = cv2.resize(image, (w-cx, cy))
                mosaic_mask[:cy, cx:] = cv2.resize(mask, (w-cx, cy))
            elif i == 2:
                mosaic_img[cy:, :cx] = cv2.resize(image, (cx, h-cy))
                mosaic_mask[cy:, :cx] = cv2.resize(mask, (cx, h-cy))
            elif i == 3:
                mosaic_img[cy:, cx:] = cv2.resize(image, (w-cx, h-cy))
                mosaic_mask[cy:, cx:] = cv2.resize(mask, (w-cx, h-cy))
        
        return mosaic_img, mosaic_mask
