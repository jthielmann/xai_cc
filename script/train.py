import os
import random

import numpy as np
import torch.nn.functional
from scipy import stats
import pandas as pd
from data_loader import get_data_loaders, get_dataset_ae, get_dataset_ae_single, get_occlusion_dataset, get_data_loader_occlusion
import json
import torch
import torch.optim as optim
import torch.nn as nn

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
             freeze_pretrained=False, pretrained_path=None,
             error_metric=lambda a, b: stats.pearsonr(a[:, 0].cpu().detach().numpy(), b[:, 0].cpu().detach().numpy())[0],
             error_metric_name="pearson corr", meta_data_dir_name="meta_data", use_default_samples=True):

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
                     'freeze_pretrained': freeze_pretrained, 'pretrained_path': str(pretrained_path)}
        json.dump(json_dict, file)

    # Defining training and validation history dictionary
    history = {'train_loss': [], 'train_corr': [], 'val_loss': [], 'val_corr': []}

    valid_loss_min = np.Inf
    valid_corr_max = np.NINF
    train_loader, val_loader = get_data_loaders(data_dir, batch_size, genes, use_default_samples=use_default_samples)

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
            model_save = model_save_dir + "/best_model.pt"
            torch.save(model.state_dict(), model_save)
        if val_corr > valid_corr_max:
            valid_corr_max = val_corr
        log_text = "AVG T Loss: {:.3f} AVG T {}: {:.3f} AVG V Loss: {:.3f} AVG V {}: {:.3f}, Best V: {:.3f}".format(
            train_loss, error_metric_name, train_corr, val_loss, error_metric_name, val_corr, valid_loss_min)
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


def train_ae(ae, out_dir_name, criterion, optimizer=None, training_data_dir="../Training_Data/", epochs=100, lr=0.001, ):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device: ", device)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # MPS not supported for now
    ae.to(device)
    if optimizer is None:
        optimizer = optim.Adam(ae.parameters(), lr=lr)

    train_loader = get_dataset_ae_single(training_data_dir, file_type="tif")
    val_loader = get_dataset_ae_single("../Training_Data/", file_type="tiff")
    best_val_loss = float('inf')
    logfile = out_dir_name + "/log.txt"
    open(logfile, "a").close()
    print("training start")
    for epoch in range(epochs):
        running_loss = 0.0
        ae.train()

        for i, data in enumerate(train_loader, 0):

            # Assume data is a tuple of (input_tensor, target_tensor)
            inputs, path = data
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(0)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = ae(inputs)

            # Compute loss
            loss = criterion(outputs, inputs)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        running_loss_val = 0.0
        ae.eval()
        #print("validation start")
        for i, data in enumerate(val_loader, 0):
            # Assume data is a tuple of (input_tensor, target_tensor)
            inputs, path = data
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(0)

            outputs = ae(inputs)

            loss = criterion(outputs, inputs)

            running_loss_val += loss.item()
        if epoch > 10 and running_loss_val < best_val_loss:
            best_val_loss = running_loss_val
            torch.save(ae.state_dict(), out_dir_name + "/best_model.pt")
        torch.save(ae.state_dict(), out_dir_name + "/latest.pt")
        f = open(logfile, "a")
        f.write(f'Epoch {epoch + 1} loss: {running_loss:.4f} val loss {running_loss_val:.4f}\n')
        f.close()
        torch.save(ae.state_dict(), "../models/" + out_dir_name + "/ep_" + str(epoch) + ".pt")

    print('Finished Training')


def train_ae2(ae, out_dir_name, criterion, optimizer=None, training_data_dir="../Training_Data/", epochs=100, lr=0.001, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device: ", device)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # MPS not supported for now
    ae.to(device)
    if optimizer is None:
        optimizer = optim.Adam(ae.parameters(), lr=lr)

    train_loader = get_data_loader_occlusion(training_data_dir, batch_size=batch_size, file_endings=["tif", "tiff"])
    val_loader = get_data_loader_occlusion("../Training_Data/", batch_size=batch_size, file_endings=["tif", "tiff"])
    best_val_loss = float('inf')
    logfile = out_dir_name + "/log.txt"
    open(logfile, "a").close()

    with open(out_dir_name + "/settings.json", "w") as file:
        json_dict = {'model_type': ae.__class__.__name__, 'criterion': str(criterion), 'batch_size': batch_size,
                     'epochs': epochs, 'optimizer': str(optimizer), 'device': device}
        json.dump(json_dict, file)

    print("training start")
    for epoch in range(epochs):
        running_loss = 0.0
        ae.train()

        for i, data in enumerate(train_loader, 0):

            # Assume data is a tuple of (input_tensor, target_tensor)
            inputs, path = data
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(0)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = ae(inputs)

            # Compute loss
            loss = criterion(outputs, inputs)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        running_loss_val = 0.0
        ae.eval()
        #print("validation start")
        for i, data in enumerate(val_loader, 0):
            # Assume data is a tuple of (input_tensor, target_tensor)
            inputs, path = data
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(0)

            outputs = ae(inputs)

            loss = criterion(outputs, inputs)

            running_loss_val += loss.item()
        if epoch > 10 and running_loss_val < best_val_loss:
            best_val_loss = running_loss_val
            torch.save(ae.state_dict(), out_dir_name + "/best_model.pt")
        torch.save(ae.state_dict(), out_dir_name + "/latest.pt")
        f = open(logfile, "a")
        f.write(f'Epoch {epoch + 1} loss: {running_loss:.4f} val loss {running_loss_val:.4f}\n')
        f.close()
        torch.save(ae.state_dict(), "../models/" + out_dir_name + "/ep_" + str(epoch) + ".pt")

    print('Finished Training')

