import lightning as L
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from script.data_processing.data_loader import get_dataset
from script.data_processing.image_transforms import get_transforms
import torch

class STDataModule(L.LightningDataModule):
    def __init__(self, genes, train_samples, val_samples, test_samples, data_dir, num_workers, use_transforms_in_model,
                 batch_size=64, debug=False, bins=1, gene_data_filename_train="gene_data.csv", gene_data_filename_val="gene_data.csv", gene_data_filename_test="gene_data.csv"):
        super().__init__()
        self.data_dir = data_dir
        if not use_transforms_in_model:
            self.transforms = get_transforms()
        else:
            self.transforms = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
        self.genes = genes
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.debug = debug
        self.bins = bins
        self.gene_data_filename_train = gene_data_filename_train
        self.gene_data_filename_val = gene_data_filename_val
        self.gene_data_filename_test = gene_data_filename_test

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            if not self.debug:
                self.train_dataset = get_dataset(self.data_dir, genes=self.genes, samples=self.train_samples, transforms=self.transforms, bins=self.bins, gene_data_filename=self.gene_data_filename_train)
                self.val_dataset = get_dataset(self.data_dir, genes=self.genes, samples=self.val_samples, transforms=self.transforms, bins=self.bins, gene_data_filename=self.gene_data_filename_val)
            else:
                self.train_dataset = get_dataset(self.data_dir, genes=self.genes, samples=self.train_samples, transforms=self.transforms, bins=self.bins, max_len=100, gene_data_filename=self.gene_data_filename_train)
                self.val_dataset = get_dataset(self.data_dir, genes=self.genes, samples=self.val_samples, transforms=self.transforms, bins=self.bins, max_len=100, gene_data_filename=self.gene_data_filename_val)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = get_dataset(self.data_dir, genes=self.genes, samples=self.test_samples, transforms=self.transforms, bins=self.bins, gene_data_filename=self.gene_data_filename_test)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False)

    def free_memory(self):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

