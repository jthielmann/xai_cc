import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import get_dataset

class STDataModule(L.LightningDataModule):
    def __init__(self, genes, train_samples, val_samples, test_samples, data_dir="../data/jonas/Training_Data/"):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # mean and std of the whole dataset
            transforms.Normalize([0.7406, 0.5331, 0.7059], [0.1651, 0.2174, 0.1574])
            ])
        self.genes = genes
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = get_dataset(self.data_dir, genes=self.genes, samples=self.train_samples, transforms=self.transforms)
            self.val_dataset = get_dataset(self.data_dir, genes=self.genes, samples=self.val_samples, transforms=self.transforms)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = get_dataset(self.data_dir, genes=self.genes, samples=self.test_samples, transforms=self.transforms)


    def train_dataloader(self, batch_size=64):
        return DataLoader(self.train_dataset, batch_size=batch_size)

    def val_dataloader(self, batch_size=64):
        return DataLoader(self.val_dataset, batch_size=batch_size)

    def test_dataloader(self, batch_size=64):
        return DataLoader(self.test_dataset, batch_size=batch_size)