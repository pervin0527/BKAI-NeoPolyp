import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from glob import glob
from torch.utils.data import Dataset


class BKAI_Dataset(Dataset):
    def __init__(self, data_dir, set_name, size=256, threshold=50, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.set_name = set_name
        self.size = (size, size)
        self.threshold = threshold
        self.transform = transform

        self.set_txt = f"{data_dir}/{set_name}.txt"
        with open(self.set_txt, "r") as f:
            file_list = f.readlines()

        self.file_list = [x.strip() for x in file_list]
        

    def __getitem__(self, index):
        file_name = self.file_list[index]
        image = cv2.imread(f"{self.data_dir}/train/{file_name}.jpeg")
        mask = cv2.imread(f"{self.data_dir}/train_gt/{file_name}.jpeg")

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]        
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1)) ## H, W, C -> C, H, W
        image = image / 255.0

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, self.size)
        mask = self.convert_mask(mask)

        return image, mask
    

    def __len__(self):
        return len(self.file_list)


    def convert_mask(self, mask):
        # label_transformed = np.full(mask.shape[:2], 2, dtype=np.uint8)  # 초기 마스크를 2(background)로 설정
        label_transformed = np.full(mask.shape[:2], 0, dtype=np.uint8)  # 초기 마스크를 0(background)로 설정

        # Red color (neoplastic polyps)를 0으로 변환
        red_mask = (mask[:, :, 0] > self.threshold) & (mask[:, :, 1] < 50) & (mask[:, :, 2] < 50)
        # label_transformed[red_mask] = 0
        label_transformed[red_mask] = 1

        # Green color (non-neoplastic polyps)를 1로 변환
        green_mask = (mask[:, :, 0] < 50) & (mask[:, :, 1] > self.threshold) & (mask[:, :, 2] < 50)
        # label_transformed[green_mask] = 1
        label_transformed[green_mask] = 2

        return label_transformed
    

if __name__ == "__main__":
    transform = A.Compose([A.Rotate(limit=35, p=0.3),
                           A.HorizontalFlip(p=0.3),
                           A.VerticalFlip(p=0.3),
                           A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    dataset = BKAI_Dataset(data_dir="/home/pervinco/Datasets/BKAI_IGH_NeoPolyp", set_name="train", transform=transform)

    for data in dataset:
        image, mask = data[0], data[1]
        image = np.transpose(image, (1, 2, 0))
        print(image.shape, mask.shape)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        
        plt.subplot(1, 2, 2)
        plt.imshow(mask)

        save_path = f"./sample.png"
        plt.savefig(save_path)
        plt.close()
        break