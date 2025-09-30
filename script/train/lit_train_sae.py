import os
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
        self.wandb_run = run
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

        # Infer d_in from the encoder output if not specified
        if "d_in" not in self.config:
            sample_batch = next(iter(self.train_loader))
            # The batch is either a tensor or a list/tuple.
            if isinstance(sample_batch, (list, tuple)):
                sample_input = sample_batch[0]
            else:
                sample_input = sample_batch
            
            with torch.no_grad():
                encoder_output = self.encoder(sample_input)
            
            d_in = encoder_output.shape[-1]
            self.config['d_in'] = d_in

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
        # Resolve device from trainer; fall back to CPU if missing
        device = getattr(self.trainer, "device", None)
        if device is None:
            device = getattr(getattr(self.trainer, "strategy", None), "root_device", None) or (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            )
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
        if self.wandb_run is not None:
            self.wandb_run.log({
                "umap_plot": wandb.Image(plt)
            })
        else:
            out_dir = self.config.get("out_path") or self.config.get("sweep_dir") or self.config.get("model_dir") or "."
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, "umap_val.png"), dpi=150, bbox_inches="tight")
        plt.close()
        
        return embedding, paths_list
