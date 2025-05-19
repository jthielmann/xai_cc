import os
import random
import numpy as np
import torch
import torch.nn.functional
import pandas as pd
from torchvision import transforms
from PIL import Image
from script.data_processing.image_transforms import get_transforms


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


class STDataset_umap(STDataset):
    def __init__(self, dataframe, image_transforms=None, device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu", inputs_only=False):
        super().__init__(dataframe, image_transforms, device, inputs_only)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        a = Image.open(row["tile"]).convert("RGB")
        if self.transforms:
            a = self.transforms(a)
        if self.device_handling:
            a = a.to(self.device)
        return a, row["tile"]


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


# contains tile path and gene data
def get_base_dataset(data_dir, genes, samples, meta_data_dir="/meta_data/", max_len=None, bins=1, gene_data_filename="gene_data.csv"):
    columns_of_interest = ["tile"]
    if genes:
        columns_of_interest += genes
    datasets = []  # Use a list to store DataFrames

    for i in samples:
        file_path = data_dir + i + meta_data_dir + gene_data_filename
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


def get_dataset(data_dir, genes, transforms=None, samples=None, meta_data_dir="/meta_data/", max_len=None, bins=1, only_inputs=False, gene_data_filename="gene_data.csv", device_handling=False):
    if samples is None:
        samples = [os.path.basename(f) for f in os.scandir(data_dir) if f.is_dir()]
    gene_data_df = get_base_dataset(data_dir, genes, samples, meta_data_dir=meta_data_dir, max_len=max_len, bins=bins, gene_data_filename=gene_data_filename)
    st_dataset = STDataset(gene_data_df, image_transforms=transforms, inputs_only=only_inputs, device_handling=device_handling)
    return st_dataset


# returns imgs and tile paths
def get_dataset_for_umap(data_dir, genes, transforms=None, samples=None, meta_data_dir="/meta_data/", max_len=None, bins=1, only_inputs=False):
    if samples is None:
        samples = [os.path.basename(f) for f in os.scandir(data_dir) if f.is_dir()]
    gene_data_df = get_base_dataset(data_dir, genes, samples, meta_data_dir=meta_data_dir, max_len=max_len, bins=bins)
    st_dataset = STDataset_umap(gene_data_df, image_transforms=transforms, inputs_only=only_inputs)
    return st_dataset


def get_dino_dataset(csv_path, dino_transforms=None, max_len=None, bins=1, device_handling=False):
    if dino_transforms is None:
        dino_transforms = get_transforms()
    file_df = pd.read_csv(csv_path, nrows=max_len)
    st_dataset = STDataset(file_df, image_transforms=dino_transforms, inputs_only=True, device_handling=device_handling)
    return st_dataset


class NCT_CRC_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, classes, use_tiles_sub_dir=False, image_transforms=None, device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu", label_as_string=True):
        self.transforms = image_transforms
        self.device = device
        self.label_as_string = label_as_string
        self.classes = classes
        self.dataframe = pd.DataFrame(columns=["tile", "class"])
        for c in classes:
            class_dir = data_dir + "/" + c
            if use_tiles_sub_dir:
                class_dir += "/tiles/"
            for file in os.scandir(class_dir):
                if file.is_file() and file.name.endswith(".tif"):
                    self.dataframe = pd.concat([self.dataframe, pd.DataFrame({"tile": [file.path], "class": [c if label_as_string else classes.index(c)]})], ignore_index=True)

    def __len__(self):
        return len(self.dataframe)

    def get_tilepath(self, index):
        return self.dataframe.iloc[index]["tile"]

    def get_class(self, index):
        return self.dataframe.iloc[index]["class"]

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img = Image.open(row["tile"]).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        return img, row["class"]

# returns imgs and tile paths
def get_dataset_NCT_CRC_classification(data_dir, transforms=None, samples=None, meta_data_dir="/meta_data/", max_len=None):
    if samples is None:
        samples = [os.path.basename(f) for f in os.scandir(data_dir) if f.is_dir()]
    st_dataset = STDataset_umap(gene_data_df, image_transforms=transforms)
    return st_dataset