import lightning as L
import numpy as np
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt
from umap import UMAP

from script.data_processing.lit_STDataModule import get_data_module
from script.model.lit_sae import LitSparseAutoencoder
from script.model.model_factory import get_encoder


class SAETrainerPipeline:
    def __init__(self, config: dict, run: wandb.sdk.wandb_run.Run):
        self.config = config
        self.run = run
        self.sae = None
        self.data_module = get_data_module(self.config)
        self.data_module.setup("fit")
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader   = self.data_module.val_dataloader()
        self.test         = self.data_module.test_dataloader()
        self.trainer = None
        self.encoder = None

    def setup(self):

        self.encoder = get_encoder(self.config.get("encoder_type"))
        self.sae = LitSparseAutoencoder(self.config, encoder=self.encoder)

        logger = None
        if self.config.get("log_to_wandb"):
            logger = WandbLogger(project=self.config["project"], name=self.config["name"])

        self.trainer = L.Trainer(
            max_epochs=self.config["epochs"],
            logger=logger,
            accelerator="auto",
            log_every_n_steps=1,
        )

    def run(self):
        if not self.trainer:
            self.setup()

        self.trainer.fit(self.sae, train_dataloaders=self.train_loader, val_dataloaders=self.val_loader)

        if self.config.get("test_samples"):
            self.trainer.test(self.sae, datamodule=self.data_module)

        if self.config.get("umap"):
            self.generate_umap()

    def generate_umap(self):
        n_components = self.config.get("umap_n_components")
        n_neighbors = self.config.get("umap_n_neighbors")

        features_list = []
        paths_list = []  # For plotting: stores original images
        device = self.trainer.device
        self.sae.to(device)
        with torch.no_grad():
            for imgs, paths in self.val_loader:
                imgs = imgs.to(device)
                # Extract features (assumes model returns feature vectors)
                features = self.sae(imgs)
                features_list.append(features.cpu().numpy())
                paths_list.extend(paths)

        # Concatenate features from all batches
        features_np = np.concatenate(features_list, axis=0)

        # Compute the UMAP embedding on the aggregated features
        umap_model = UMAP(n_components=n_components, random_state=42, n_neighbors=n_neighbors)
        embedding = umap_model.fit_transform(features_np)

        plt.figure(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], s=1)
        plt.title("UMAP of Validation Set Features")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        
        self.run.log({
            "umap_plot": wandb.Image(plt)
        })
        plt.close()
        
        return embedding, paths_list