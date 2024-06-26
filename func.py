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


class MyNet(nn.Module):
    def __init__(self, my_pretrained_model):
        super(MyNet, self).__init__()
        self.pretrained = my_pretrained_model
        self.my_new_layers = nn.Sequential(nn.Linear(1000, 200),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(200),
                                           nn.Dropout(0.3),
                                           nn.Linear(200, 10),  #choose x in nn.Linear(20,x) depending on n_classes
                                           nn.ReLU(),
                                           nn.Dropout(0.3))

        self.gene1 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

    def forward(self, x):
        x = self.pretrained(x)
        x = self.my_new_layers(x)
        gene1 = self.gene1(x)
        return gene1


def get_model():
    extended_net = MyNet(my_pretrained_model=models.resnet50(weights="IMAGENET1K_V2"))

    # Initializing weights for newly appended layers in MyNet
    def initialize_weights_linear(model_to_initialize):
        for m in model_to_initialize.modules():
            if isinstance(m, nn.Linear):
                n = m.in_features
                y = 1.0 / np.sqrt(n)
                m.weight.data.uniform_(-y, y)
                m.bias.data.fill_(0)

    extended_net.apply(initialize_weights_linear)

    print(extended_net)  # prints the final model architecture
    return extended_net


def train_epoch(resnet, device, dataloader, criterion, optimizer):
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
        loss = criterion(g_output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        corr = stats.pearsonr(g_output[:, 0].cpu().detach().numpy(), labels[:, 0].cpu().detach().numpy())[0]
        batch_corr_train += corr
        once = False

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
            loss = criterion(g_output, labels)
            valid_loss += loss.item()

            corr = stats.pearsonr(g_output[:, 0].cpu().detach().numpy(), labels[:, 0].cpu().detach().numpy())[
                0]
            batch_corr_val += corr

    return valid_loss, batch_corr_val


def log_training(date, training_log):
    with open(training_log, "a") as f:
        f.write(
            date + " Resnet50 - single gene\top: AdamW\telrs: 0.9\tlfn: MSE Loss\n")  # Adapt to model and gene name(s) getting trained


def my_batch_iter(data_loader):
    data, labels = next(iter(data_loader))
    return data, labels


# adapted implementation from : https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
def load_images_from_dir(dir_path):
    images = []
    for filename in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, filename))[:, :, ::-1]
        if img is not None:
            trans = np.transpose(img, (2, 0, 1))
            images.append(trans)

    return images


class CustomImageDataset(Dataset):
    def __init__(self, data_dir):
        self.dirs = []
        for name in os.listdir(data_dir):
            if not os.path.isdir(name):
                continue
            self.dirs.append(name)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        image = self.dirs[idx]
        label = self.dirs[idx]
        return image, label


def load_dataset(data_dir, gene, columns_of_interest):
    patients = [f.path for f in os.scandir(data_dir) if f.is_dir()]

    # generate training dataframe with all training samples
    for i in patients:
        adata = ad.read_h5ad(data_dir + i + "/Preprocessed_STDataset/tmm_combat_scaled_" + i + ".h5ad")
        st_dataset = adata.to_df()
        st_dataset["tile"] = st_dataset.index

        st_dataset_filtered = st_dataset.copy()
        st_dataset_filtered = st_dataset_filtered[columns_of_interest]
        if i == "p007":
            st_dataset_filtered['tile'] = st_dataset_filtered['tile'].str.replace('-0',
                                                                                  '')  # remove unwanted path strings
        elif i == "p014":
            st_dataset_filtered['tile'] = st_dataset_filtered['tile'].str.replace('-4', '')
        elif i == "p016":
            st_dataset_filtered['tile'] = st_dataset_filtered['tile'].str.replace('-5', '')
        elif i == "p020":
            st_dataset_filtered['tile'] = st_dataset_filtered['tile'].str.replace('-6', '')
        elif i == "p025":
            st_dataset_filtered['tile'] = st_dataset_filtered['tile'].str.replace('-8', '')

        st_dataset_filtered['tile'] = st_dataset_filtered['tile'].apply(
            lambda x: "{}{}{}{}".format(data_dir, i, "/Tiles_156/",
                                        x[57:]))  # adjust path to load directory

        train_st_dataset = pd.concat([train_st_dataset, st_dataset_filtered])  # concat all samples


class STDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        gene_names = list(self.dataframe)[1:]
        gene_vals = []
        row = self.dataframe.iloc[index]
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.7406, 0.5331, 0.7059], [0.1651, 0.2174, 0.1574])
            # apply normalization transforms as for pretrained colon classifier
        ])
        a = Image.open(row["tile"]).convert("RGB")
        # print(x.size)
        for j in gene_names:
            gene_val = float(row[j])
            gene_vals.append(gene_val)
        e = row["tile"]
        a = transform(a)
        return a, gene_vals, e


class Subset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    # get image and label tensor from dataset and transform
    def __getitem__(self, index):
        a, gene_vals, e = self.subset[index]
        if self.transform:
            a = self.transform(a)
        return (a, gene_vals, e)

    # get length of dataset
    def __len__(self):
        return len(self.subset)


def get_patient_loader(data_dir, batch_size, patient=None, gene="RUBCNL"):
    columns_of_interest = ["tile", gene]

    train_st_dataset = pd.DataFrame(columns=columns_of_interest)

    # generate training dataframe with all training samples
    st_dataset = pd.read_csv(data_dir + patient + "/Preprocessed_STDataset/gene_data.csv", index_col=-1)
    st_dataset["tile"] = st_dataset.index
    st_dataset['tile'] = st_dataset['tile'].apply(lambda x: str(data_dir) + "/" + str(patient) + "/Tiles_156/" + str(x))

    train_st_dataset = pd.concat([train_st_dataset, st_dataset[columns_of_interest]])  # concat all samples

    train_st_dataset.reset_index(drop=True, inplace=True)
    loaded_train_dataset = STDataset(train_st_dataset)
    """
    # Training and validation transforms (i.e. train data augmentation)
    train_transforms = transforms.RandomApply([transforms.RandomRotation(degrees=
                                                                         (0, 180)),
                                               transforms.RandomHorizontalFlip(
                                                   p=0.75),
                                               transforms.RandomVerticalFlip(
                                                   p=0.75)], p=0.5)

    train_data = Subset(loaded_train_dataset, transform=train_transforms)
    """
    #val_data = Subset(loaded_valid_dataset, transform=train_transforms)
    #train_loader = DataLoader(dataset=loaded_train_dataset, batch_size=batch_size, shuffle=False)
    return loaded_train_dataset


def get_data_loaders(data_dir, batch_size, gene="RUBCNL"):
    train_samples = ["p007", "p014", "p016", "p020", "p025"]
    val_samples = ["p009", "p013"]  # 8 21 26 page 42

    columns_of_interest = ["tile", gene]

    train_st_dataset = pd.DataFrame(columns=columns_of_interest)
    valid_st_dataset = pd.DataFrame(columns=columns_of_interest)

    # generate training dataframe with all training samples
    for i in train_samples:
        st_dataset = pd.read_csv(data_dir + i + "/Preprocessed_STDataset/gene_data.csv", index_col=-1)
        print(st_dataset.head())
        st_dataset["tile"] = st_dataset.index
        print(st_dataset.head())
        st_dataset['tile'] = st_dataset['tile'].apply(lambda x: str(data_dir) + "/" + str(i) + "/Tiles_156/" + str(x))

        train_st_dataset = pd.concat([train_st_dataset, st_dataset[columns_of_interest]])  # concat all samples

    # generate validation dataframe with all validation samples
    for i in val_samples:

        st_dataset = pd.read_csv(data_dir + i + "/Preprocessed_STDataset/gene_data.csv")
        st_dataset["tile"] = st_dataset.index
        st_dataset['tile'] = st_dataset['tile'].apply(lambda x: str(data_dir) + "/" + str(i) + "/Tiles_156/" + str(x))

        valid_st_dataset = pd.concat([valid_st_dataset, st_dataset[columns_of_interest]])

    # reset index of dataframes
    train_st_dataset.reset_index(drop=True, inplace=True)
    valid_st_dataset.reset_index(drop=True, inplace=True)



    loaded_train_dataset = STDataset(train_st_dataset)
    loaded_valid_dataset = STDataset(valid_st_dataset)

    # Training and validation transforms (i.e. train data augmentation)
    train_transforms = transforms.RandomApply([transforms.RandomRotation(degrees=
                                                                         (0, 180)),
                                               transforms.RandomHorizontalFlip(
                                                   p=0.75),
                                               transforms.RandomVerticalFlip(
                                                   p=0.75)], p=0.5)
    # transforms.RandomAdjustSharpness(
    # sharpness_factor = 2, p = 0.75),
    # transforms.RandomAutocontrast(p = 0.75),
    # transforms.ColorJitter(hue=(-0.2, 0.2))
    # ], p = 0.5)

    train_data = Subset(loaded_train_dataset, transform=train_transforms)

    #val_data = Subset(loaded_valid_dataset, transform=train_transforms)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=loaded_train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


def training(resnet, data_dir, epochs, loss_fn, learning_mode, batch_size, gene):
    print(getframeinfo(currentframe()).lineno)

    data_path = Path(data_dir)
    training_log = f"./results/ST_Predict_absolute_single.txt"
    date = str(datetime.today().strftime('%d%m%Y'))
    log_training(date, training_log)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("device: ", device)
    resnet.to(device)
    # Defining gradient function
    learning_rate = 0.000005 if learning_mode == "LLR" else 0.00001 if learning_mode == "NLR" else 0.0005
    optimizer = optim.AdamW([{"params": resnet.pretrained.parameters(), "lr": learning_rate},
                             {"params": resnet.my_new_layers.parameters(), "lr": learning_rate}], weight_decay=0.005)

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
        print("train_epoch")
        training_loss, training_corr = train_epoch(resnet, device, train_loader, loss_fn, optimizer)
        print("valid_epoch")
        valid_loss, valid_corr = valid_epoch(resnet, device, val_loader, loss_fn)

        train_loss = (training_loss / len(train_loader.dataset)) * 1000
        val_loss = (valid_loss / len(val_loader.dataset)) * 1000
        train_corr = training_corr / len(train_loader)
        val_corr = valid_corr / len(val_loader)

        if val_loss < valid_loss_min or val_corr > valid_corr_max:
            valid_loss_min = val_loss
            model_save = data_dir + date + "_single_" + "_" + str(learning_rate) + "resnet.pt"
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
