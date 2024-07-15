import os
import cv2
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
from scipy import stats
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import pandas as pd
import anndata as ad
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# debug
from inspect import currentframe, getframeinfo

DEFAULT_RANDOM_SEED = 42
print(getframeinfo(currentframe()).lineno)


def seed_basic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def seed_torch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_everything(seed=DEFAULT_RANDOM_SEED):
    seed_basic(seed)
    seed_torch(seed)


seed_everything(seed=DEFAULT_RANDOM_SEED)


def train_epoch(resnet, device, dataloader, criterion, optimizer, epoch):
    train_loss = 0.0
    batch_corr_train = 0.0
    resnet.train()
    i = 0
    for images, labels, path in dataloader:
        if i % 20 == 0:
            print("train_epoch iteration ", i)
        i += 1
        images = images.to(device)
        images = images.float()

        labels = torch.stack(labels, dim=1)
        labels = labels.float()
        labels = labels.to(device)
        optimizer.zero_grad()
        g_output = resnet(images)
        loss = criterion(g_output.squeeze(), labels[:, 0])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        corr = stats.pearsonr(g_output[:, 0].cpu().detach().numpy(), labels[:, 0].cpu().detach().numpy())[0]
        #print(g_output.shape, labels.shape)
        #print(corr, batch_corr_train)
        batch_corr_train += corr

    return train_loss, batch_corr_train


def valid_epoch(resnet, device, dataloader, criterion):
    valid_loss = 0.0
    batch_corr_val = 0.0
    resnet.eval()
    with torch.no_grad():
        i = 0
        for images, labels, path in dataloader:
            if i % 20 == 0:
                print("valid_epoch iteration ", i)
            i += 1
            images = images.to(device)
            images = images.float()

            labels = torch.stack(labels, dim=1)
            labels = labels.float()
            labels = labels.to(device)
            g_output = resnet(images)
            loss = criterion(g_output, labels[:, 0])
            valid_loss += loss.item()

            corr = stats.pearsonr(g_output[:, 0].cpu().detach().numpy(), labels[:, 0].cpu().detach().numpy())[
                0]
            batch_corr_val += corr

    return valid_loss, batch_corr_val


def log_training(date, training_log):
    with open(training_log, "a") as f:
        f.write(
            date + " Resnet50 - single gene\top: AdamW\telrs: 0.9\tlfn: MSE Loss\n")  # Adapt to model and gene name(s) getting trained


def training(resnet, data_dir, model_save_dir, epochs, loss_fn, learning_mode, batch_size, gene):
    print(getframeinfo(currentframe()).lineno)

    training_log = f"../results/ST_Predict_absolute_single.txt"
    date = str(datetime.today().strftime('%d%m%Y'))
    log_training(date, training_log)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("device: ", device)
    resnet.to(device)
    # Defining gradient function
    learning_rate = 0.000005 if learning_mode == "LLR" else 0.00001 if learning_mode == "NLR" else 0.0005


    optimizer = optim.AdamW([{"params": resnet.pretrained.parameters(), "lr": learning_rate},
                             #{"params": resnet.my_new_layers.parameters(), "lr": learning_rate},
                             {"params": resnet.gene1.parameters(), "lr": learning_rate}], weight_decay=0.005)




    #optimizer = optim.AdamW([{"params": resnet.pretrained.parameters(), "lr": learning_rate}], weight_decay=0.005)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    #Defining training and validation history dictionary
    history = {'train_loss': [], 'train_corr': [], 'val_loss': [], 'val_corr': []}
    valid_loss_min = np.Inf
    valid_corr_max = np.NINF

    #Iterate through epochs
    for epoch in range(epochs):
        print('Epoch {} / {}:'.format(epoch + 1, epochs))

        epoch_to_print = "Epoch {} / {}:".format(epoch + 1, epochs)
        with open(training_log, "a") as f:
            f.write(epoch_to_print + "\n")

        #Load data into Dataloader
        print("getting loaders")
        train_loader, val_loader = get_data_loaders(data_dir, batch_size, gene)
        print(len(train_loader))
        print("train_epoch")
        training_loss, training_corr = train_epoch(resnet, device, train_loader, loss_fn, optimizer, epoch)
        print("valid_epoch")
        valid_loss, valid_corr = valid_epoch(resnet, device, val_loader, loss_fn)

        train_loss = (training_loss / len(train_loader.dataset)) * 1000
        val_loss = (valid_loss / len(val_loader.dataset)) * 1000
        train_corr = training_corr / len(train_loader)
        val_corr = valid_corr / len(val_loader)

        if val_loss < valid_loss_min:
            valid_loss_min = val_loss
        if val_corr > valid_corr_max:
            valid_corr_max = val_corr
        model_save = model_save_dir + date + "_ep_" + str(epoch) + "_lr_" + str(learning_rate) + "resnet.pt"
        torch.save(resnet.state_dict(), model_save)

        # Save training log into text file
        log_to_print = "AVG T Loss:{:.3f} AVG T Correlation:{:.3f} AVG V Loss:{:.3f} AVG V Correlation:{:.3f}".format(
            train_loss, train_corr, val_loss, val_corr)
        print(log_training(date, training_log))
        with open(training_log, "a") as f:
            f.write(log_to_print)
            f.write("\n")

        # Get training and validation history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_corr'].append(train_corr)
        history['val_corr'].append(val_corr)

        scheduler.step()
    history_df = pd.DataFrame.from_dict(history, orient="columns")

    save_name = (data_dir + date + "_single_" + str(batch_size) + "_" + str(learning_rate) + "_train_history.csv")
    history_df.to_csv(save_name, index=False)
