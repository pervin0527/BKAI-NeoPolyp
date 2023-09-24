import cv2
import random
import numpy as np
import albumentations as A

from torch.utils.data import Dataset

class BKAIDataset(Dataset):
    def __init__(self, base_dir, split, size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.base_dir = base_dir
        self.split = split
        self.size = size
        self.mean, self.std = mean, std

        self.set_txt = f"{base_dir}/{split}.txt"
        with open(self.set_txt, "r") as f:
            file_list = f.readlines()

        self.file_list = [x.strip() for x in file_list]

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
        

    def __getitem__(self, index):
        image, mask = self.load_img_mask(index)

        if self.split == "train":
            # transform_image, transform_mask = self.train_img_mask_transform(image, mask)
            prob = random.random()
            if prob < 0.3:
                transform_image, transform_mask = self.train_img_mask_transform(image, mask)
            elif 0.3 < prob < 0.6:
                transform_image, transform_mask = self.mosaic_augmentation(image, mask)
            else:
                transform_image, transform_mask = self.cutmix_augmentation(image, mask)

            if random.random() > 0.5:
                transform_image = self.train_image_transform(image)
        
        else:
            transform_image = image
            transform_mask = mask
        
        transform_image = self.normalize(transform_image)
        transform_mask = self.encode_mask(transform_mask)

        return transform_image, transform_mask
    

    def __len__(self):
        return len(self.file_list)


    def normalize(self, image):
        image = np.array(image).astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std

        image = np.transpose(image, (2, 0, 1)) ## H, W, C -> C, H, W

        return image

    
    def load_img_mask(self, index):
        file_name = self.file_list[index]
        image = cv2.imread(f"{self.base_dir}/train/{file_name}.jpeg")
        mask = cv2.imread(f"{self.base_dir}/train_mask/{file_name}.jpeg") 
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size))

        return image, mask


    def encode_mask(self, mask):
        label_transformed = np.zeros(shape=mask.shape[:-1], dtype=np.uint8)

        red_mask = mask[:, :, 0] >= 100
        label_transformed[red_mask] = 1

        green_mask = mask[:, :, 1] >= 100
        label_transformed[green_mask] = 2

        return label_transformed
    

    def train_img_mask_transform(self, image, mask):     
        transformed = self.train_transform(image=image, mask=mask)
        transformed_image, transformed_mask = transformed["image"], transformed["mask"]

        return transformed_image, transformed_mask
    

    def mosaic_augmentation(self, image, mask):
        h, w = self.size, self.size
        mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)
        mosaic_mask = np.zeros((h, w, 3), dtype=np.uint8)
        cx, cy = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)

        candidates = []
        is_full = False
        while not is_full:
            idx = random.randint(0, len(self.file_list)-1)

            if not idx in candidates:
                candidates.append(idx)

            if len(candidates) < 4:
                is_full = True
        
        indices = [0, 1, 2, 3]
        random.shuffle(indices)
        for i, index in enumerate(indices):
            image, mask = self.load_img_mask(index)
            
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


    def train_image_transform(self, image):     
        transformed = self.image_transform(image=image)
        transformed_image = transformed["image"]

        return transformed_image
    

    def cutmix_augmentation(self, image, mask):
        idx = random.randint(0, len(self.file_list) - 1)
        mix_image, mix_mask = self.load_img_mask(idx)

        lam = np.clip(np.random.beta(1.0, 1.0), 0.2, 0.8)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.shape, lam)

        image[bbx1:bbx2, bby1:bby2] = mix_image[bbx1:bbx2, bby1:bby2]
        mask[bbx1:bbx2, bby1:bby2] = mix_mask[bbx1:bbx2, bby1:bby2]

        return image, mask


    def rand_bbox(self, size, lam):
        W = size[1]
        H = size[0]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int64(W * cut_rat)
        cut_h = np.int64(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2