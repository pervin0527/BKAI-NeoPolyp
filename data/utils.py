import cv2
import torch
import random
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

def get_color_labels():
    return np.asarray([[0, 0, 0], [255, 0, 0], [0, 255, 0]])


def decode_segmap(label_mask, plot=False):
    n_classes = 3
    label_colours = get_color_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0

    if plot:
        plt.imshow(rgb)
        plt.show()

    else:
        return rgb


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample["image"]
        mask = sample["mask"]
        
        image = np.array(image).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        image /= 255.0
        image -= self.mean
        image /= self.std

        return {"image" : image, "mask" : mask}
    

class ToTensor(object):
    def __call__(self, sample):
        image = sample["image"]
        mask = sample["mask"]

        image = np.array(image).astype(np.float32).transpose((2, 0, 1)) # numpy image: H x W x C --> torch image: C X H X W
        mask = np.array(mask).astype(np.float32)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return {'image': image, 'mask': mask}


def valid_augmentation(sample):
    transform = A.Compose([A.Resize(256, 256, p=1)])
    
    image, mask = np.array(sample["image"]), np.array(sample["mask"])
    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']

    return {"image" : transformed_image, "mask" : transformed_mask}


def train_augmentation(sample):
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
                                              p=0.7)], p=1),
                            ])
    
    image, mask = np.array(sample["image"]), np.array(sample["mask"])
    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']

    return {"image" : transformed_image, "mask" : transformed_mask}


def mosaic_augmentation(samples, output_size=(256, 256)):
    h, w = output_size
    mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)
    mosaic_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 중심점 지정
    cx, cy = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
    
    indices = [0, 1, 2, 3]
    random.shuffle(indices)
    for i, index in enumerate(indices):
        image, mask = np.array(samples[index]["image"]), np.array(samples[index]["mask"])
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
    
    return {"image" : mosaic_img, "mask" : mosaic_mask}
