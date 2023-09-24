import os
import yaml
import torch

from tqdm import tqdm
from utils import make_test_txt
from model.TransResUNet import TResUnet

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min([os.cpu_count(), config["batch_size"] if config["batch_size"] > 1 else 0, 8])

    data_dir = config["data_dir"]
    test = config["test"]
    if os.path.exists(f"{data_dir}/{test}.txt"):
        make_test_txt(data_dir, test)

    model = TResUnet(backbone=config["backbone"], num_layers=config["num_layers"])
    model = model.to(device)

    saved_weights = torch.load(config["pretrain_weight"])
    model.load_state_dict(saved_weights, strict=False)