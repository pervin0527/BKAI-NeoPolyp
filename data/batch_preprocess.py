import cv2
import random
import numpy as np

def load_img_mask(image_path, mask_path=None, size=256, only_img=False):
    if only_img:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size, size))

        return image
    
    else:
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path) 
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (size, size))
        mask = cv2.resize(mask, (size, size))

        return image, mask


def encode_mask(mask):
    label_transformed = np.zeros(shape=mask.shape[:-1], dtype=np.uint8)

    green_mask = mask[:, :, 1] >= 100
    label_transformed[green_mask] = 1

    red_mask = mask[:, :, 0] >= 100
    label_transformed[red_mask] = 2

    return label_transformed


def normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = np.array(image).astype(np.float32)
    image /= 255.0
    image -= mean
    image /= std

    image = np.transpose(image, (2, 0, 1)) ## H, W, C -> C, H, W

    return image


def train_img_mask_transform(transform, image, mask):     
    transformed = transform(image=image, mask=mask)
    transformed_image, transformed_mask = transformed["image"], transformed["mask"]

    return transformed_image, transformed_mask


def train_image_transform(transform, image):     
    transformed = transform(image=image)
    transformed_image = transformed["image"]

    return transformed_image


def mosaic_augmentation(piecies, size):
    h, w = size, size
    mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)
    mosaic_mask = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
    
    indices = [0, 1, 2, 3]
    random.shuffle(indices)
    for i, index in enumerate(indices):
        image, mask = piecies[index][0], piecies[index][1]
        
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


def rand_bbox(size, lam):
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


def cutmix_augmentation(image1, mask1, image2, mask2):
    lam = np.clip(np.random.beta(1.0, 1.0), 0.2, 0.8)
    bbx1, bby1, bbx2, bby2 = rand_bbox(image1.shape, lam)

    image1[bbx1:bbx2, bby1:bby2] = image2[bbx1:bbx2, bby1:bby2]
    mask1[bbx1:bbx2, bby1:bby2] = mask2[bbx1:bbx2, bby1:bby2]

    return image1, mask1