import os
import cv2
import yaml
import torch
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from model.TransResUNet import TResUnet
from data.batch_preprocess import normalize
from utils import decode_mask


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 0] = 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded

    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]

    return rle_to_string(rle)


def rle2mask(mask_rle, shape=(3,3)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape).T


def mask2string(dir):
    ## mask --> string
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        
        img = cv2.imread(path)[:, :, ::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:,:,channel])
            strings.append(string)

    r = {'ids': ids, 'strings': strings,}

    return r


def test(model, files, dir):
    for idx in tqdm(range(len(files))):
        file = files[idx]
        file_name = file.split('/')[-1].split('.')[0]

        image = cv2.imread(file)
        height, width, channel = image.shape

        image = cv2.resize(image, (config["img_size"], config["img_size"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        x = normalize(image, mean=config["mean"], std=config["std"])
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x).to(device)

        y_pred = model(x)
        y_pred = torch.argmax(y_pred[0], dim=0).cpu().numpy()

        decoded_mask = decode_mask(y_pred)
        decoded_mask = cv2.resize(decoded_mask, (width, height))
        decoded_mask = cv2.cvtColor(decoded_mask, cv2.COLOR_BGR2RGB)

        cv2.imwrite(f"{dir}/{file_name}.png", decoded_mask)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min([os.cpu_count(), config["batch_size"] if config["batch_size"] > 1 else 0, 8])

    data_dir = config["data_dir"]
    test_dir = config["test"]
    files = sorted(glob(f"{data_dir}/{test_dir}/*"))

    folder = "/".join(config["test_weight"].split('/')[:-2])
    folder = f"{folder}/prediction"
    if not os.path.isdir(folder):
        os.makedirs(folder)

    model = TResUnet(backbone=config["backbone"], num_layers=config["num_layers"])
    model = model.to(device)
    model.load_state_dict(torch.load(config["test_weight"]), strict=False)

    test(model, files, folder)

    result = mask2string(folder)
    df = pd.DataFrame(columns=['Id', 'Expected'])
    df['Id'] = result['ids']
    df['Expected'] = result['strings']

    df.to_csv(r'output.csv', index=False)