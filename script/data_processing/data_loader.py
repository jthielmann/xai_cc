import os
import random
from idlelib.pyparse import trans

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from script.data_processing.custom_transforms import Occlude
from script.data_processing.image_transforms import get_transforms

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
    def __init__(self, dataframe, image_transforms=None, device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu", inputs_only=False, device_handling=True):
        self.dataframe = dataframe
        self.transforms = image_transforms
        self.device = device
        self.device_handling = device_handling
        self.only_inputs = inputs_only

    def __len__(self):
        return len(self.dataframe)

    def get_tilename(self, index):
        return self.dataframe.iloc[index]["tile"]

    def __getitem__(self, index):
        gene_names = list(self.dataframe)[:-1]
        row = self.dataframe.iloc[index]
        a = Image.open(row["tile"]).convert("RGB")
        # print(x.size)
        if self.transforms:
            a = self.transforms(a)
        if self.device_handling:
            a = a.to(self.device)

        # clustering uses crp which currently only supports classification tasks. therefore we take class 0 as a hack
        # as we only have one gene output in clustering because we cluster each gene separately
        if self.only_inputs:
            return a, 0
        gene_vals = []
        for j in gene_names:
            #gene_vals.append(float(row[j]))
            #mps:
            gene_val = torch.tensor(float(row[j]), dtype=torch.float32)
            #gene_val = torch.tensor(float(row[j]))
            gene_vals.append(gene_val)
        return a, torch.stack(gene_vals)



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
        a, gene_vals = self.subset[index]
        if self.transform:
            a = self.transform(a)
        return a, gene_vals

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


# contains tile path and gene data
def get_base_dataset(data_dir, genes, samples, meta_data_dir="/meta_data/", max_len=None, bins=1):
    columns_of_interest = ["tile"]
    if genes:
        columns_of_interest += genes
    datasets = []  # Use a list to store DataFrames

    for i in samples:
        file_path = data_dir + i + meta_data_dir + "gene_data.csv"
        st_dataset_patient = pd.read_csv(file_path, usecols=columns_of_interest, nrows=max_len)

        st_dataset_patient["tile"] = [os.path.join(data_dir, i, "tiles", tilename) for tilename in st_dataset_patient.tile]

        datasets.append(st_dataset_patient)

    st_dataset = pd.concat(datasets)

    # resample data to have equal number of samples in each bin
    if bins > 1:
        # group dataset by gene values into #bins groups
        gene_values = st_dataset[genes[0]]
        gene_value_bins = pd.cut(gene_values, bins)
        grouped_dataset = st_dataset.groupby(gene_value_bins)

        # sample from each group
        sample_size = grouped_dataset.size().max()
        sampled_dataset = grouped_dataset.apply(lambda x: x.sample(sample_size, replace=True) if len(x) > 0 else x)
        st_dataset = sampled_dataset.reset_index(drop=True)

        #st_dataset = st_dataset.groupby(pd.cut(st_dataset[genes[0]], bins)).apply(lambda x: x.sample(st_dataset.shape[0] // bins, replace=True)).reset_index(drop=True)
    return st_dataset


def get_dataset(data_dir, genes, transforms=None, samples=None, meta_data_dir="/meta_data/", max_len=None, bins=1, only_inputs=False):
    if samples is None:
        samples = [os.path.basename(f) for f in os.scandir(data_dir) if f.is_dir()]
    gene_data_df = get_base_dataset(data_dir, genes, samples, meta_data_dir=meta_data_dir, max_len=max_len, bins=bins)
    st_dataset = STDataset(gene_data_df, image_transforms=transforms, inputs_only=only_inputs)
    return st_dataset


def get_data_loaders(data_dir, batch_size, genes, use_default_samples=True, meta_data_dir="/meta_data/", samples=None):
    if use_default_samples:
        train_samples = get_train_samples()
        val_samples = get_val_samples()
    elif samples:
        train_samples = samples[0]
        val_samples = samples[1]
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
        st_dataset = pd.read_csv(data_dir + i + meta_data_dir + "/gene_data.csv", index_col=-1)
        st_dataset["tile"] = st_dataset.index
        st_dataset['tile'] = st_dataset['tile'].apply(lambda x: str(data_dir) + "/" + str(i) + "/tiles/" + str(x))
        if train_st_dataset.empty:
            train_st_dataset = st_dataset[columns_of_interest]
        else:
            train_st_dataset = pd.concat([train_st_dataset, st_dataset[columns_of_interest]])  # concat all samples

    # generate validation dataframe with all validation samples
    for i in val_samples:
        st_dataset = pd.read_csv(data_dir + i + meta_data_dir + "/gene_data.csv", index_col=-1)
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

    print("train_samples", train_samples)
    print("val_samples", val_samples)

    return train_loader, val_loader

class PlottingDataset(torch.utils.data.Dataset):
    from script.data_processing.image_transforms import get_transforms
    def __init__(self, dataframe, device,
                 transforms=get_transforms()):
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
def get_dataset_for_plotting(data_dir, genes, samples=None, device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"):
    if samples is None:
        samples = []
        for subdir in os.listdir(data_dir):
            if os.path.isdir(data_dir + "/" + subdir):
                samples.append(subdir)

    columns_of_interest = ["tile"]
    for gene in genes:
        columns_of_interest.append(gene)
    dataset = pd.DataFrame(columns=columns_of_interest)

    # generate training dataframe with all training samples
    for i in samples:
        st_dataset = pd.read_csv(data_dir + "/" + i + "/meta_data/gene_data.csv", index_col=-1)
        st_dataset["tile"] = st_dataset.index
        st_dataset["tile"] = st_dataset["tile"].apply(lambda x: str(data_dir) + "/" + str(i) + "/tiles/" + str(x))
        if dataset.empty:
            dataset = st_dataset[columns_of_interest]
        else:
            # concat
            dataset = pd.concat([dataset, st_dataset[columns_of_interest]])

    # reset index of dataframes
    dataset.reset_index(drop=True, inplace=True)

    return PlottingDataset(dataset, device=device)


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


def get_occlusion_dataset(data_dir, mean=None, std=None, file_endings=None):
    if not mean:
        mean = [0.7406, 0.5331, 0.7059]
    if not std:
        std = [0.1651, 0.2174, 0.1574]
    dataset = CustomImageDataset(data_dir, transforms=transforms.Compose([transforms.Resize((224, 224)),
                                                                                 transforms.ToTensor(),
                                                                                 transforms.Normalize(mean, std)]), file_endings=file_endings)
    return dataset


def get_data_loader_occlusion(data_dir, batch_size, mean=None, std=None, file_endings=None):
    dataset = get_occlusion_dataset(data_dir, mean, std, file_endings=file_endings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def get_dino_dataset(csv_path, transforms=get_transforms(), max_len=None, bins=1, device_handling=False):
    file_df = pd.read_csv(csv_path, nrows=max_len)
    st_dataset = STDataset(file_df, image_transforms=transforms, inputs_only=True, device_handling=device_handling)
    return st_dataset