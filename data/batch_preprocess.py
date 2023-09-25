import cv2
import copy
import random
import numpy as np
import albumentations as A
from scipy.ndimage import label

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
    # image -= mean
    # image /= std

    image = np.transpose(image, (2, 0, 1)) ## H, W, C -> C, H, W

    return image


def train_img_mask_transform(transform, image, mask): 
    x, y = copy.deepcopy(image), copy.deepcopy(mask)
    transformed = transform(image=x, mask=y)
    transformed_image, transformed_mask = transformed["image"], transformed["mask"]

    return transformed_image, transformed_mask


def train_image_transform(transform, image):
    x = copy.deepcopy(image)     
    transformed = transform(image=x)
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
        piece_image, piece_mask = piecies[index][0], piecies[index][1]
        
        if i == 0:
            mosaic_img[:cy, :cx] = cv2.resize(piece_image, (cx, cy))
            mosaic_mask[:cy, :cx] = cv2.resize(piece_mask, (cx, cy))
        elif i == 1:
            mosaic_img[:cy, cx:] = cv2.resize(piece_image, (w-cx, cy))
            mosaic_mask[:cy, cx:] = cv2.resize(piece_mask, (w-cx, cy))
        elif i == 2:
            mosaic_img[cy:, :cx] = cv2.resize(piece_image, (cx, h-cy))
            mosaic_mask[cy:, :cx] = cv2.resize(piece_mask, (cx, h-cy))
        elif i == 3:
            mosaic_img[cy:, cx:] = cv2.resize(piece_image, (w-cx, h-cy))
            mosaic_mask[cy:, cx:] = cv2.resize(piece_mask, (w-cx, h-cy))
    
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


def crop_colors_from_mask_and_image(original_image, original_mask, margin=1):
    image, mask = copy.deepcopy(original_image), copy.deepcopy(original_mask)
    red_mask = ((mask[:, :, 0] >= 200) & (mask[:, :, 0] <= 255)).astype(int)
    green_mask = ((mask[:, :, 1] >= 200) & (mask[:, :, 1] <= 255)).astype(int)
    
    labeled_red, num_red = label(red_mask)
    labeled_green, num_green = label(green_mask)
    
    red_crops, green_crops = [], []
    for i in range(1, num_red + 1):
        y, x = np.where(labeled_red == i)
        y_min, y_max, x_min, x_max = y.min(), y.max(), x.min(), x.max()
        
        cropped_img = image[y_min-margin:y_max+margin, x_min-margin:x_max+margin].copy()
        cropped_mask = mask[y_min-margin:y_max+margin, x_min-margin:x_max+margin].copy()
        
        red_crops.append((cropped_img, cropped_mask))
    
    for i in range(1, num_green + 1):
        y, x = np.where(labeled_green == i)
        y_min, y_max, x_min, x_max = y.min(), y.max(), x.min(), x.max()
        
        cropped_img = image[y_min-margin:y_max+margin, x_min-margin:x_max+margin].copy()
        cropped_mask = mask[y_min-margin:y_max+margin, x_min-margin:x_max+margin].copy()
        
        green_crops.append((cropped_img, cropped_mask))
    
    return red_crops, green_crops


def mixup(crops, image, mask, mixup_times=2, alpha=0.5, threshold=200):
    base_image, base_mask = copy.deepcopy(image), copy.deepcopy(mask)
    piece_transform = A.Compose([A.RandomRotate90(p=1, always_apply=True),
                                 A.HorizontalFlip(p=0.7),
                                 A.VerticalFlip(p=0.7)])
    
    base_transform = A.Compose([A.RandomRotate90(p=1, always_apply=True),
                                A.HorizontalFlip(p=0.7),
                                A.VerticalFlip(p=0.7)])   

    B_height, B_width, _ = base_image.shape
    mixup_track_mask = np.zeros((B_height, B_width))
    
    for _ in range(mixup_times):
        for crop in crops:
            crop_image, crop_mask = crop[0], crop[1]

            piece_transformed = piece_transform(image=crop_image, mask=crop_mask)
            t_piece_image, t_piece_mask = piece_transformed["image"], piece_transformed["mask"]
            height, width, _ = t_piece_image.shape
                
            max_attempts = 1000
            for _ in range(max_attempts):
                i, j = random.randint(0, B_height - height - 1), random.randint(0, B_width - width - 1)
                region_mask = base_mask[i:i+height, j:j+width]
                region_track = mixup_track_mask[i:i+height, j:j+width]
                
                if (region_mask[:, :, :3].sum(axis=2) <= threshold).all() and not region_track.any():
                    base_mask[i:i+height, j:j+width] = region_mask * alpha + t_piece_mask * (1 - alpha)
                    region_image = base_image[i:i+height, j:j+width]
                    base_image[i:i+height, j:j+width] = region_image * alpha + t_piece_image * (1 - alpha)
                    mixup_track_mask[i:i+height, j:j+width] = 1
                    break
            
    base_transformed = base_transform(image=base_image, mask=base_mask)
    t_base_image, t_base_mask = base_transformed["image"], base_transformed["mask"]
        
    return t_base_image, t_base_mask
