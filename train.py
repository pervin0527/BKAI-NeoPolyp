import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import yaml
import torch

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.TransResUNet import TResUnet
from data.BKAIDataset import BKAIDataset
from metric.losses import DiceLoss
from metric.scores import MultiClassDiceScore
from utils import epoch_time, predict, save_config_to_yaml, print_and_save, save_visualization


def eval(model, dataloader, loss_fn, acc_fn, device):
    model.eval()

    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for idx, (origin_x, origin_y, batch_x, batch_y) in enumerate(tqdm(dataloader, desc="Valid", unit="batch")):
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device, dtype=torch.float32)

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.long())
            epoch_loss += loss.item()

            acc = acc_fn(y_pred, batch_y.long())
            epoch_acc += acc.item()

    epoch_loss = epoch_loss / len(dataloader)
    epoch_acc = epoch_acc / len(dataloader)

    return epoch_loss, epoch_acc


def train(epoch, model, dataloader, optimizer, loss_fn, acc_fn, device, fig_dir):
    model.train()

    epoch_loss = 0
    epoch_acc = 0
    for idx, (origin_x, origin_y, batch_x, batch_y) in enumerate(tqdm(dataloader, desc="Train", unit="batch")):
        # save_visualization(epoch, idx, origin_x, origin_y, batch_x, batch_y, fig_dir)

        batch_x = batch_x.to(device, dtype=torch.float32)
        batch_y = batch_y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(batch_x)
        loss = loss_fn(y_pred, batch_y.long())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        acc = acc_fn(y_pred, batch_y.long())
        epoch_acc += acc.item()

    epoch_loss = epoch_loss / len(dataloader)
    epoch_acc = epoch_acc / len(dataloader)

    return epoch_loss, epoch_acc


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ## Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min([os.cpu_count(), config["batch_size"] if config["batch_size"] > 1 else 0, 8])

    ## Load Dataset
    train_dataset = BKAIDataset(config=config, split=config["train"])
    valid_dataset = BKAIDataset(config=config, split=config["valid"])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config["batch_size"], num_workers=num_workers)

    ## Load pre-trained weight & models
    model = TResUnet(backbone=config["backbone"], num_layers=config["num_layers"])
    model = model.to(device)

    if config["pretrain_weight"] != "":
        saved_weights = torch.load(config["pretrain_weight"])

        keys_to_remove = []
        for key in saved_weights.keys():
            if 'output' in key:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            saved_weights.pop(key)

        model.load_state_dict(saved_weights, strict=False)

    ## Loss Function
    loss_fn = DiceLoss(crossentropy=config["crossentropy"])
    acc_fn = MultiClassDiceScore()

    ## Optimizer & LR Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["initial_lr"], betas=config["betas"], weight_decay=config["weight_decay"])
    if config["scheduler"] == "decay":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=config["patience"], verbose=True)

    elif config["scheduler"] == "onecycle":
        div_factor = config["max_lr"] / config["initial_lr"]
        final_div_factor = config["max_lr"] / config["initial_lr"]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        anneal_strategy="cos",
                                                        max_lr=config["max_lr"],
                                                        total_steps=config["epochs"],
                                                        pct_start=config["pct_start"],
                                                        div_factor=div_factor,
                                                        final_div_factor=final_div_factor,
                                                        verbose=True)


    ## make save dir
    save_dir = config["save_dir"]
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = f"{save_dir}/{current_time}"

    if not os.path.isdir(save_path):
        print(save_path)
        config["save_dir"] = save_path
        os.makedirs(f"{save_path}/weights")
        os.makedirs(f"{save_path}/predict")
        os.makedirs(f"{save_path}/logs")
        os.makedirs(f"{save_path}/batch")
    
    save_config_to_yaml(config, save_path)


    ## Train start
    print("\nTrain Start.")
    writer = SummaryWriter(log_dir=f"{save_path}/logs")
    
    early_stopping_count = 0
    patience = config["early_stopping_patience"]

    best_valid_loss = float("inf")
    epochs = config["epochs"]
    for epoch in range(epochs):
        start_time = time.time()

        current_lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train(epoch, model, train_dataloader, optimizer, loss_fn, acc_fn, device, f"{save_path}/batch")
        valid_loss, valid_acc = eval(model, valid_dataloader, loss_fn, acc_fn, device)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Valid", valid_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Valid", valid_acc, epoch)        
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        data_str = f"Epoch [{epoch+1:02}/{epochs}] | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tCurrent Learning Rate: {current_lr} \n"
        data_str += f"\tTrain Loss: {train_loss:.4f} | Train_Acc: {train_acc:.4f} \n"
        data_str += f"\tValid Loss: {valid_loss:.4f} | Valid_Acc: {valid_acc:.4f} \n"

        if valid_loss < best_valid_loss:
            data_str += f"\tLoss decreased. {best_valid_loss:.4f} ---> {valid_loss:.4f} \n"
            # print_and_save(f"{save_path}/logs", data_str)
            best_valid_loss = valid_loss

            torch.save(model.state_dict(), f"{save_path}/weights/best.pth")
            early_stopping_count = 0

        elif valid_loss > best_valid_loss:
            data_str += f"\tLoss not decreased. {best_valid_loss:.4f} Remaining patience: [{early_stopping_count}/{patience}] \n"
            early_stopping_count += 1

        if config["scheduler"] == "decay":
            scheduler.step(valid_loss)
        elif config["scheduler"] == "onecycle":
            scheduler.step()

        # print_and_save(f"{save_path}/logs", data_str)
        # predict(epoch + 1, config, model=model, device=device)

        if early_stopping_count == config["early_stopping_patience"]:
            data_str = f"Early stopping: validation loss stops improving from last {patience} continously.\n"
            # print_and_save(f"{save_path}/logs", data_str)
            break

        print_and_save(f"{save_path}/logs/train_log.txt", data_str)
        predict(epoch + 1, config, model=model, device=device)
    
    writer.close()