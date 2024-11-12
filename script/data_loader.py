import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from custom_transforms import Occlude


# debug
from inspect import currentframe, getframeinfo


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
    #seed_basic(seed)
    seed_torch(seed)


seed_everything(seed=DEFAULT_RANDOM_SEED)


def log_training(date, training_log):
    with open(training_log, "a") as f:
        f.write(
            date + " Resnet50 - single gene\top: AdamW\telrs: 0.9\tlfn: MSE Loss\n")  # Adapt to model and gene name(s) getting trained


class STDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transforms=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # mean and std of the whole dataset
            transforms.Normalize([0.7406, 0.5331, 0.7059], [0.1651, 0.2174, 0.1574])
            ])):
        self.dataframe = dataframe
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        gene_names = list(self.dataframe)[1:]
        gene_vals = []
        row = self.dataframe.iloc[index]
        a = Image.open(row["tile"]).convert("RGB")
        # print(x.size)
        for j in gene_names:
            gene_val = float(row[j])
            gene_vals.append(gene_val)
        e = row["tile"]
        # apply normalization transforms as for pretrained colon classifier
        a = self.transforms(a)
        return a, gene_vals, e


class TileLoader:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # mean and std of the whole dataset
            transforms.Normalize([0.7406, 0.5331, 0.7059], [0.1651, 0.2174, 0.1574])
            ])

    def open(self, path):
        a = Image.open(path).convert("RGB")
        return self.transforms(a)


class Subset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    # get image and label tensor from dataset and transform
    def __getitem__(self, index):
        a, gene_vals, e = self.subset[index]
        if self.transform:
            a = self.transform(a)
        return a, gene_vals, e

    # get length of dataset
    def __len__(self):
        return len(self.subset)


def get_patient_loader(data_dir, patient, genes=None):

    if genes is None:
        columns_of_interest = ["tile", "RUBCNL"]
    else:
        columns_of_interest = ["tile"]
        for gene in genes:
            columns_of_interest.append(gene)
    train_st_dataset = pd.DataFrame(columns=columns_of_interest)

    # generate training dataframe with all training samples
    st_dataset = pd.read_csv(data_dir + patient + "/meta_data/gene_data.csv", index_col=-1)
    st_dataset["tile"] = st_dataset.index
    st_dataset['tile'] = st_dataset['tile'].apply(lambda x: str(data_dir) + "/" + str(patient) + "/tiles/" + str(x))

    if train_st_dataset.empty:
        train_st_dataset = st_dataset[columns_of_interest]
    else:
        train_st_dataset = pd.concat([train_st_dataset, st_dataset[columns_of_interest]])  # concat all samples

    train_st_dataset.reset_index(drop=True, inplace=True)
    loaded_train_dataset = STDataset(train_st_dataset)
    return loaded_train_dataset


def get_train_samples():
    return ["p007", "p014", "p016", "p020", "p025"]


def get_val_samples():
    return ["p009", "p013"]


def get_test_samples():
    return ["p008", "p021", "p026"]


def get_data_loaders(data_dir, batch_size, genes, use_default_samples=True):
    if use_default_samples:
        train_samples = get_train_samples()
        val_samples = get_val_samples()
    else:
        print("setting train and val samples automatically")
        patients = [os.path.basename(f) for f in os.scandir(data_dir) if f.is_dir()]
        pop_ids = []
        for i in range(len(patients)):
            if patients[i].startswith("."):
                pop_ids.append(i)
        pop_ids.sort(reverse=True)
        for i in pop_ids:
            patients.pop(i)
        train_sample_count = int(0.5 + len(patients) * 0.8)  # 80% of dataset is train, +0.5 means we round
        train_samples = patients[0:train_sample_count]
        val_samples   = patients[train_sample_count:]
        print("train_samples: ")
        print(*patients[0:train_sample_count])
        print("val_samples: ")
        print(*patients[train_sample_count:])

    columns_of_interest = ["tile"]
    for gene in genes:
        columns_of_interest.append(gene)
    train_st_dataset = pd.DataFrame(columns=columns_of_interest)
    valid_st_dataset = pd.DataFrame(columns=columns_of_interest)

    # generate training dataframe with all training samples
    for i in train_samples:
        st_dataset = pd.read_csv(data_dir + i + "/meta_data/gene_data.csv", index_col=-1)
        st_dataset["tile"] = st_dataset.index
        st_dataset['tile'] = st_dataset['tile'].apply(lambda x: str(data_dir) + "/" + str(i) + "/tiles/" + str(x))
        if train_st_dataset.empty:
            train_st_dataset = st_dataset[columns_of_interest]
        else:
            train_st_dataset = pd.concat([train_st_dataset, st_dataset[columns_of_interest]])  # concat all samples

    # generate validation dataframe with all validation samples
    for i in val_samples:
        st_dataset = pd.read_csv(data_dir + i + "/meta_data/gene_data.csv", index_col=-1)
        st_dataset["tile"] = st_dataset.index
        st_dataset['tile'] = st_dataset['tile'].apply(lambda x: str(data_dir) + "/" + str(i) + "/tiles/" + str(x))

        if valid_st_dataset.empty:
            valid_st_dataset = st_dataset[columns_of_interest]
        else:
            valid_st_dataset = pd.concat([valid_st_dataset, st_dataset[columns_of_interest]])  # concat all samples

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

    train_data = Subset(loaded_train_dataset, transform=train_transforms)
    valid_data = Subset(loaded_valid_dataset, transform=train_transforms)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class PlottingDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe,
                 device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
                 transforms=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # mean and std of the whole dataset
            transforms.Normalize([0.7406, 0.5331, 0.7059], [0.1651, 0.2174, 0.1574])
            ])):
        self.dataframe = dataframe
        self.transforms = transforms
        self.device = device

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        gene_names = list(self.dataframe)[1:]
        gene_vals = []
        row = self.dataframe.iloc[index]
        a = Image.open(row["tile"]).convert("RGB")
        # print(x.size)
        for j in gene_names:
            gene_val = float(row[j])
            gene_vals.append(gene_val)
        e = row["tile"]
        # apply normalization transforms as for pretrained colon classifier
        a = self.transforms(a)
        a = a.to(self.device)
        return a, 0


# has the labels set to 0 because that makes it easier to work with the frameworks written for classification
# the idea is that they filter the attribution by the chosen class, but as we only have one output we always choose y=0
def get_dataset_for_plotting(data_dir, gene="RUBCNL", samples=None):
    if samples is None:
        samples = ["p007", "p014", "p016", "p020", "p025"]

    columns_of_interest = ["tile", gene]
    dataset = pd.DataFrame(columns=columns_of_interest)

    # generate training dataframe with all training samples
    for i in samples:
        st_dataset = pd.read_csv(data_dir + "/" + i + "/meta_data/gene_data.csv", index_col=-1)
        st_dataset["tile"] = st_dataset.index
        st_dataset['tile'] = st_dataset['tile'].apply(lambda x: str(data_dir) + "/" + str(i) + "/tiles/" + str(x))
        if dataset.empty:
            dataset = st_dataset[columns_of_interest]
        else:
            # concat
            dataset = pd.concat([dataset, st_dataset[columns_of_interest]])

    # reset index of dataframes
    dataset.reset_index(drop=True, inplace=True)

    return PlottingDataset(dataset)


class ae_dataset(torch.utils.data.Dataset):
    def __init__(self, data, transforms_= transforms.Compose([transforms.Resize((224, 224))])):
        if data is None:
            raise ValueError("ae_dataset __init__: no images provided")
        self.transforms = transforms_
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transforms(self.data[index][0]), self.data[index][1]


# has the labels set to 0 because that makes it easier to work with the frameworks written for classification
# the idea is that they filter the attribution by the chosen class, but as we only have one output we always choose y=0
def get_dataset_ae(data_dir, val_data_dir=None, file_type="tif"):
    print("setting train and val samples automatically")
    patients = [os.path.basename(f) for f in os.scandir(data_dir) if f.is_dir()]
    pop_ids = []
    # remove hidden files
    for i in range(len(patients)):
        if patients[i].startswith("."):
            pop_ids.append(i)
    pop_ids.sort(reverse=True)
    for i in pop_ids:
        patients.pop(i)

    if val_data_dir is not None:
        train_sample_count = int(0.5 + len(patients) * 0.8)  # 80% of dataset is train, +0.5 means we round
    else:
        train_sample_count = len(patients)
    patients_train = patients[0:train_sample_count]
    if val_data_dir is None:
        patients_val = patients[train_sample_count:]
    else:
        patients_val = [os.path.basename(f) for f in os.scandir(val_data_dir) if f.is_dir()]
        pop_ids = []
        # remove hidden files
        for i in range(len(patients_val)):
            if patients_val[i].startswith("."):
                pop_ids.append(i)
        pop_ids.sort(reverse=True)
        for i in pop_ids:
            patients_val.pop(i)
    print("train_samples: ")
    print(patients_train)
    print("val_samples: ")
    print(patients_val)
    columns=["tile, path"]

    transform = transforms.ToTensor()

    dataset = []
    # generate training dataframe with all training samples
    for i in patients_train:
        patient_path = data_dir + "/" + i + "/tiles/"
        for img in os.listdir(patient_path):
            if not img.endswith("." + file_type):
                continue
            dataset.append((transform(Image.open(patient_path + img).convert("RGB")), data_dir + "/" + i + "/tiles/" + img))
    dataset_train = ae_dataset(dataset)

    dataset = []
    # generate training dataframe with all training samples
    file_type = "tiff"
    for i in patients_val:
        patient_path = val_data_dir + "/" + i + "/tiles/"
        for img in os.listdir(patient_path):
            if not img.endswith("." + file_type):
                continue
            dataset.append((transform(Image.open(patient_path + img).convert("RGB")), data_dir + "/" + i + "/tiles/" + img))
    dataset_val = ae_dataset(dataset)
    # reset index of dataframes

    return dataset_train, dataset_val


def get_dataset_ae_single(data_dir, file_type="tif", squelch=True):
    patients = [os.path.basename(f) for f in os.scandir(data_dir) if f.is_dir()]
    pop_ids = []

    # remove hidden files
    for i in range(len(patients)):
        if patients[i].startswith("."):
            pop_ids.append(i)
    # reversed so that we dont reduce the length of the list when we remove an element
    pop_ids.sort(reverse=True)
    for i in pop_ids:
        patients.pop(i)

    train_sample_count = len(patients)
    patients_train = patients[0:train_sample_count]

    transform = transforms.ToTensor()
    dataset = []
    # generate training dataframe with all training samples
    for i in patients_train:
        if not squelch:
            print(i)
        patient_path = data_dir + "/" + i + "/tiles/"
        for img in os.listdir(patient_path):
            if not img.endswith("." + file_type):
                continue
            dataset.append((transform(Image.open(patient_path + img).convert("RGB")), data_dir + "/" + i + "/tiles/" + img))
    dataset_loaded = ae_dataset(dataset)
    return dataset_loaded


def get_dataset_ae_split(data_dir, split=0.8, file_type="tif"):
    print("setting train and val samples automatically")
    patients = [os.path.basename(f) for f in os.scandir(data_dir) if f.is_dir()]
    pop_ids = []
    # remove hidden files
    for i in range(len(patients)):
        if patients[i].startswith("."):
            pop_ids.append(i)
    pop_ids.sort(reverse=True)
    for i in pop_ids:
        patients.pop(i)

    if split != 0.0:
        train_sample_count = int(0.5 + len(patients) * split)  # 80% of dataset is train, +0.5 means we round
    else:
        train_sample_count = len(patients)
    patients_train = patients[0:train_sample_count]

    if split is None:
        patients_val = patients[train_sample_count:]
    else:
        patients_val = [os.path.basename(f) for f in os.scandir(val_data_dir) if f.is_dir()]
        pop_ids = []
        # remove hidden files
        for i in range(len(patients_val)):
            if patients_val[i].startswith("."):
                pop_ids.append(i)
        pop_ids.sort(reverse=True)
        for i in pop_ids:
            patients_val.pop(i)
    print("train_samples: ")
    print(patients_train)
    print("val_samples: ")
    print(patients_val)
    columns = ["tile, path"]

    transform = transforms.ToTensor()

    dataset = []
    # generate training dataframe with all training samples
    for i in patients_train:
        patient_path = data_dir + "/" + i + "/tiles/"
        for img in os.listdir(patient_path):
            if not img.endswith("." + file_type):
                continue
            dataset.append(
                (transform(Image.open(patient_path + img).convert("RGB")), data_dir + "/" + i + "/tiles/" + img))
    dataset_train = ae_dataset(dataset)

    dataset = []
    # generate training dataframe with all training samples
    file_type = "tiff"
    for i in patients_val:
        patient_path = val_data_dir + "/" + i + "/tiles/"
        for img in os.listdir(patient_path):
            if not img.endswith("." + file_type):
                continue
            dataset.append(
                (transform(Image.open(patient_path + img).convert("RGB")), data_dir + "/" + i + "/tiles/" + img))
    dataset_val = ae_dataset(dataset)
    # reset index of dataframes

    return dataset_train, dataset_val


class CustomImageDataset(Dataset):
    def __init__(self, data_dir, genes=None, transforms=None, file_endings=None):
        if file_endings is None:
            file_endings = ["tif", "tiff"]
        """
        Args:
            root_dir (string): Directory with all the images organized into class folders.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = data_dir
        self.transforms = transforms
        patients = os.listdir(data_dir)
        pop_ids = []
        for p in patients:
            if p.startswith("."):
                pop_ids.append(patients.index(p))
        pop_ids.sort(reverse=True)
        for i in pop_ids:
            patients.pop(i)
        self.patients = patients

        self.imgs = []
        self.labels = []
        self.tilenames = []
        for idx, patient in enumerate(self.patients):
            patient_path = data_dir + "/" + patient
            if not os.path.isdir(patient_path):
                continue
            if genes:
                gene_data = pd.read_csv(patient_path + "/meta_data/gene_data.csv")

            for img_file in os.listdir(patient_path + "/tiles/"):
                if not img_file.endswith(".tif") and not img_file.endswith(".tiff"):
                    continue
                img_path = patient_path + "/tiles/" + img_file
                for ending in file_endings:
                    if img_file.find(ending) == -1:
                        continue
                if img_file.startswith("."):
                    continue
                labels = []
                if genes:
                    row = gene_data[gene_data["tile"] == os.path.basename(img_path)]
                    if row.empty:
                        continue
                    for gene in genes:
                        labels.append(row[gene].values[0])
                self.labels.append(labels)
                self.imgs.append(Image.open(img_path).convert("RGB"))
                self.tilenames.append(img_path)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.transforms:
            img = self.transforms(img)
        if len(self.labels) != 0:
            label = self.labels[idx]
            return img, label
        return img


def get_occlusion_dataset(data_dir, mean=None, std=None, occluder=None, file_endings=None):
    if not mean:
        mean = [0.7406, 0.5331, 0.7059]
    if not std:
        std = [0.1651, 0.2174, 0.1574]
    if not occluder:
        occluder = Occlude(20, 20, 0)
    dataset = CustomImageDataset(data_dir, transforms=transforms.Compose([transforms.Resize((224, 224)),
                                                                                 transforms.ToTensor(),
                                                                                 transforms.Normalize(mean, std),
                                                                                 occluder]), file_endings=file_endings)
    return dataset


def get_data_loader_occlusion(data_dir, batch_size, mean=None, std=None, occluder=None, file_endings=None):
    dataset = get_occlusion_dataset(data_dir, mean, std, occluder, file_endings=file_endings)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
