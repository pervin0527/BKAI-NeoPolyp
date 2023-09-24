import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from data.utils import ToTensor, Normalize, train_augmentation, valid_augmentation, mosaic_augmentation, decode_segmap

class BKAIDatasetV2(Dataset):
    NUM_CLASSES = 3

    def __init__(self, base_dir, threshold=50, split="train", mean=None, std=None):
        super().__init__()
        self.split = split
        self.threshold = int(threshold)
        self.mean = [float(x) for x in mean]
        self.std = [float(x) for x in std]

        self.base_dir = base_dir
        self.image_dir = f"{base_dir}/train"
        self.mask_dir = f"{base_dir}/train_gt"

        with open(f"{self.base_dir}/{self.split}.txt", "r") as f:
            lines = f.read().splitlines()

        self.images, self.masks = [], []
        for idx, line in enumerate(lines):
            image = f"{self.image_dir}/{line}.jpeg"
            mask = f"{self.mask_dir}/{line}.jpeg"

            self.images.append(image)
            self.masks.append(mask)

        print(f"Number of images in {self.split} : {len(self.images)}")


    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, index):
        image, mask = self.get_image_mask_pair(index)
        sample = {"image" : image, "mask" : mask}

        if self.split == "train":
            if random.random() > 0.5:
                sample = train_augmentation(sample)

            else:
                samples = [sample]
                for _ in range(3):
                    i = random.randint(0, len(self.images)-1)
                    image_i, mask_i = self.get_image_mask_pair(i)
                    samples.append({"image" : image_i, "mask" : mask_i})
                
                sample = mosaic_augmentation(samples)

        else:
            sample = valid_augmentation(sample)


        sample["mask"] = self.encode_segmask(sample["mask"])        
        # return sample
        return self.tensor_transform(sample)

    
    def get_image_mask_pair(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("RGB")

        return image, mask
    

    def encode_segmask(self, mask):
        mask = np.array(mask)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)

        red_mask = (mask[:, :, 0] > self.threshold) & (mask[:, :, 1] < 50) & (mask[:, :, 2] < 50)
        label_mask[red_mask] = 1

        green_mask = (mask[:, :, 0] < 50) & (mask[:, :, 1] > self.threshold) & (mask[:, :, 2] < 50)
        label_mask[green_mask] = 2

        return label_mask


    def tensor_transform(self, sample):
        transform = transforms.Compose([Normalize(mean=self.mean, std=self.std),
                                        ToTensor()])
        
        return transform(sample)
    

if __name__ == "__main__":
    train_dataset = BKAIDatasetV2("/home/pervinco/Datasets/BKAI_IGH_NeoPolyp", 50, "train")
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    if not os.path.isdir("./samples"):
        os.makedirs("./samples")

    for ii, sample in enumerate(train_dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['mask'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp)
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            ax1.imshow(img_tmp)
            ax1.set_title("Image")
            ax1.axis('off')
            
            
            ax2.imshow(segmap, cmap='gray')
            ax2.set_title("Mask")
            ax2.axis('off')

            plt.savefig(f"./samples/sample_{jj}.png", bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

        if ii == 1:
            break