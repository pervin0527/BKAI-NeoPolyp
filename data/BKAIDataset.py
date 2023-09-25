import random
import albumentations as A

from torch.utils.data import Dataset
from data.batch_preprocess import *

class BKAIDataset(Dataset):
    def __init__(self, config, split):
        super().__init__()
        self.split = split
        self.size = config["img_size"]
        self.augment = config["augment"]
        self.base_dir = config["data_dir"]
        self.mean, self.std = config["mean"], config["std"]

        self.train_transform = A.Compose([A.Rotate(limit=90, border_mode=0, p=0.6),
                                          A.HorizontalFlip(p=0.7),
                                          A.VerticalFlip(p=0.7)])   
        
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

        if self.split == "train" and self.augment:
            prob = random.random()
            if prob < 0.3:
            # if prob < 0.5:
                transform_image, transform_mask = train_img_mask_transform(self.train_transform, image, mask)

            elif 0.3 < prob < 0.7:
            # elif 0.5 < prob:
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

            elif 0.7 < prob < 1:
                try:
                    r_crops, g_crops = crop_colors_from_mask_and_image(image, mask, margin=1)
                    total_crops = r_crops + g_crops
                    transform_image, transform_mask = mixup(total_crops, image, mask)
                except:
                    transform_image, transform_mask = train_img_mask_transform(self.train_transform, image, mask)
        
            batch_image = normalize(transform_image, self.mean, self.std)
            batch_mask = encode_mask(transform_mask)

            return image, mask, batch_image, batch_mask
        
        else:
            batch_image = normalize(image, self.mean, self.std)
            batch_mask = encode_mask(mask)

            return image, mask, batch_image, batch_mask