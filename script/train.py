import os
import random

import numpy as np
import torch.nn.functional
from scipy import stats
import pandas as pd
from data_loader import get_data_loaders
import json
from model import get_Resnet_ae


DEFAULT_RANDOM_SEED = 42

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


def train_epoch(model, device, dataloader, criterion, optimizer, freeze_pretrained, error_metric):
    train_loss = 0.0
    batch_corr_train = 0.0
    model.train()
    if freeze_pretrained:
        model.pretrained.eval()
        for param in model.pretrained.parameters():
            param.required_grad = False
    for images, labels, path in dataloader:
        images = images.to(device)
        images = images.float()

        labels = torch.stack(labels, dim=1)
        labels = labels.float()
        labels = labels.to(device)
        optimizer.zero_grad()
        g_output = model(images)
        loss = criterion(g_output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        corr = error_metric(g_output, labels)
        batch_corr_train += corr

    return train_loss, batch_corr_train


def valid_epoch(model, device, dataloader, criterion, error_metric):
    valid_loss = 0.0
    batch_corr_val = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels, path in dataloader:
            images = images.to(device)
            images = images.float()

            labels = torch.stack(labels, dim=1)
            labels = labels.float()
            labels = labels.to(device)
            g_output = model(images)
            loss = criterion(g_output, labels)
            valid_loss += loss.item()

            corr = error_metric(g_output, labels)
            batch_corr_val += corr

    return valid_loss, batch_corr_val


def training(model, data_dir, model_save_dir, epochs, loss_fn, optimizer, learning_rate, batch_size, gene,
             freeze_pretrained=False, error_metric=
             lambda a, b: stats.pearsonr(a[:, 0].cpu().detach().numpy(), b[:, 0].cpu().detach().numpy())[0],
             error_metric_name="pearson corr"):

    training_log = model_save_dir + "/log.txt"
    open(training_log, "a").close()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("device: ", device)
    print("gene:", gene)
    model.to(device)

    # Defining gradient function
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    with open(model_save_dir + "/settings.json", "w") as file:
        json_dict = {'model_type': str(model.__class__.__name__), 'loss_fn': str(loss_fn), 'learning_rate': learning_rate, 'batch_size': batch_size, 'gene': gene,
                   'epochs': epochs, 'optimizer': str(optimizer), 'scheduler': str(scheduler), 'device': device, 'freeze_pretrained': freeze_pretrained}
        json.dump(json_dict, file)

    # Defining training and validation history dictionary
    history = {'train_loss': [], 'train_corr': [], 'val_loss': [], 'val_corr': []}

    valid_loss_min = np.Inf
    valid_corr_max = np.NINF
    train_loader, val_loader = get_data_loaders(data_dir, batch_size, gene)

    # Iterate through epochs
    for epoch in range(epochs):
        print('Epoch {} / {}:'.format(epoch + 1, epochs))

        epoch_to_print = "Epoch {} / {}:".format(epoch + 1, epochs)
        with open(training_log, "a") as f:
            f.write(epoch_to_print + "\n")

        # Load data into Dataloader
        print("train")
        training_loss, training_corr = train_epoch(model, device, train_loader, loss_fn, optimizer,
                                                   freeze_pretrained, error_metric)
        train_loss = (training_loss / len(train_loader.dataset)) * 1000
        train_corr = training_corr / len(train_loader)
        print("valid")
        valid_loss, valid_corr = valid_epoch(model, device, val_loader, loss_fn, error_metric)
        val_loss = (valid_loss / len(val_loader.dataset)) * 1000
        val_corr = valid_corr / len(val_loader)
        if val_loss < valid_loss_min:
            valid_loss_min = val_loss
        if val_corr > valid_corr_max:
            valid_corr_max = val_corr
        log_text = "AVG T Loss: {:.3f} AVG T {}: {:.3f} AVG V Loss: {:.3f} AVG V {}: {:.3f}".format(
            train_loss, error_metric_name, train_corr, val_loss, error_metric_name, val_corr)
        print(log_text)
        history['val_loss'].append(val_loss)
        history['val_corr'].append(val_corr)

        history['train_loss'].append(train_loss)
        history['train_corr'].append(train_corr)
        if (epoch + 1) % 10 == 0:
            model_save = model_save_dir + str(model.__class__.__name__) + "_ep_" + str(epoch) + ".pt"
            torch.save(model.state_dict(), model_save)

        # Save training log into text file
        with open(training_log, "a") as f:
            f.write(log_text)
            f.write("\n")

        scheduler.step()
    history_df = pd.DataFrame.from_dict(history, orient="columns")

    save_name = (model_save_dir + "/train_history.csv")
    history_df.to_csv(save_name, index=False)


def training_multi(model, data_dir, model_save_dir, epochs, loss_fn, optimizer, learning_rate, batch_size, genes,
             freeze_pretrained=False, error_metric=
             lambda a, b: stats.pearsonr(a[:, 0].cpu().detach().numpy(), b[:, 0].cpu().detach().numpy())[0],
             error_metric_name="pearson corr"):

    print("genes:", genes)

    training_log = model_save_dir + "/log.txt"
    open(training_log, "a").close()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    print("device: ", device)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # Defining gradient function
    with open(model_save_dir + "/settings.json", "w") as file:
        json_dict = {'model_type': model.model_type, 'random_weights': model.random_weights,
                     'dropout': model.dropout, 'drop_out_rate': model.dropout_value, 'pretrained_out_dim': str(model.pretrained_out_dim),
                     'loss_fn': str(loss_fn), 'learning_rate': learning_rate, 'batch_size': batch_size, 'genes': genes,
                     'epochs': epochs, 'optimizer': str(optimizer), 'scheduler': str(scheduler), 'device': device,
                     'freeze_pretrained': freeze_pretrained}
        json.dump(json_dict, file)

    # Defining training and validation history dictionary
    history = {'train_loss': [], 'train_corr': [], 'val_loss': [], 'val_corr': []}

    valid_loss_min = np.Inf
    valid_corr_max = np.NINF
    train_loader, val_loader = get_data_loaders(data_dir, batch_size, genes)

    # Iterate through epochs
    for epoch in range(epochs):
        print('Epoch {} / {}:'.format(epoch + 1, epochs))

        epoch_to_print = "Epoch {} / {}:".format(epoch + 1, epochs)
        with open(training_log, "a") as f:
            f.write(epoch_to_print + "\n")

        # Load data into Dataloader
        print("train")
        training_loss, training_corr = train_epoch(model, device, train_loader, loss_fn, optimizer,
                                                   freeze_pretrained, error_metric)
        train_loss = (training_loss / len(train_loader.dataset)) * 1000
        train_corr = training_corr / len(train_loader)
        print("valid")
        valid_loss, valid_corr = valid_epoch(model, device, val_loader, loss_fn, error_metric)
        val_loss = (valid_loss / len(val_loader.dataset)) * 1000
        val_corr = valid_corr / len(val_loader)
        if val_loss < valid_loss_min:
            valid_loss_min = val_loss
        if val_corr > valid_corr_max:
            valid_corr_max = val_corr
        log_text = "AVG T Loss: {:.3f} AVG T {}: {:.3f} AVG V Loss: {:.3f} AVG V {}: {:.3f}".format(
            train_loss, error_metric_name, train_corr, val_loss, error_metric_name, val_corr)
        print(log_text)
        history['val_loss'].append(val_loss)
        history['val_corr'].append(val_corr)

        history['train_loss'].append(train_loss)
        history['train_corr'].append(train_corr)
        if (epoch + 1) % 10 == 0:
            model_save = model_save_dir + "/ep_" + str(epoch) + ".pt"
            torch.save(model.state_dict(), model_save)

        # Save training log into text file
        with open(training_log, "a") as f:
            f.write(log_text)
            f.write("\n")

        scheduler.step()
    history_df = pd.DataFrame.from_dict(history, orient="columns")

    save_name = (model_save_dir + "/train_history.csv")
    history_df.to_csv(save_name, index=False)


import torch
import torch.optim as optim
import torch.nn as nn


# Define a simple training loop function
def train_decoder(model, criterion, optimizer, device, genes=None):
    if genes is None:
        genes = ["RUBCNL"]
    model.train()  # Set the model to training mode
    train_loader, val_loader = get_data_loaders("../Training_Data/", 64, genes)
    for epoch in range(40):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            # Assume data is a tuple of (input_tensor, target_tensor)
            inputs, _, path = data
            inputs = inputs.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, inputs)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

            # Print statistics every 100 mini-batches
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        model.save("../models/ae_double_resnet18/ep_" + str(epoch) + ".pt")

    print('Finished Training')



# Initialize the decoder, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
ae = get_Resnet_ae().to(device)

# Loss function and optimizer
criterion = nn.MSELoss()  # Assuming we're using mean squared error for image generation
optimizer = optim.Adam(ae.parameters(), lr=0.001)

# Train the decoder for 10 epochs
train_decoder(ae, criterion, optimizer, device)

