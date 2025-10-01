import os
import lightning as L
import numpy as np
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import pandas as pd
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
        print("sae run debug")
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
        paths_list = []
        
        batch_size = self.config['batch_size']
        dataset = self.data_module.val_dataset

        device = getattr(self.trainer, "device", None)
        if device is None:
            device = getattr(getattr(self.trainer, "strategy", None), "root_device", None) or (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            )
        self.sae.to(device)
        with torch.no_grad():
            for batch_idx, imgs in enumerate(self.val_loader):
                imgs = imgs.to(device)
                features = self.encoder(imgs)
                features = self.sae(features)
                features_list.append(features.cpu().numpy())
                
                start_idx = batch_idx * batch_size
                end_idx = start_idx + len(imgs)
                for i in range(start_idx, end_idx):
                    paths_list.append(dataset.get_tilename(i))

        features_np = np.concatenate(features_list, axis=0)

        if features_np.ndim == 3:
            features_np = features_np.reshape(features_np.shape[0], -1)

        umap_model = UMAP(n_components=n_components, random_state=42, n_neighbors=n_neighbors)
        embedding = umap_model.fit_transform(features_np)

        out_dir = self.config.get("out_path") or self.config.get("sweep_dir") or self.config.get("model_dir") or "."
        os.makedirs(out_dir, exist_ok=True)

        # --- Plot 1: Original Scatter Plot ---
        plt.figure(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], s=1)
        plt.title("UMAP of Validation Set Features")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        if self.wandb_run is not None:
            self.wandb_run.log({"umap_plot": wandb.Image(plt)})
        else:
            plt.savefig(os.path.join(out_dir, "umap_val.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # --- Plot 2: UMAP Colored by Patient ---
        patient_ids = dataset.df['patient'].tolist()
        unique_patients = sorted(list(set(patient_ids)))
        # Use a color map that provides distinct colors
        cmap = plt.get_cmap('tab20', len(unique_patients))
        patient_to_color = {p: cmap(i) for i, p in enumerate(unique_patients)}
        
        plt.figure(figsize=(14, 10))
        for patient in unique_patients:
            idx = [i for i, p in enumerate(patient_ids) if p == patient]
            if not idx: continue
            plt.scatter(embedding[idx, 0], embedding[idx, 1], s=25, color=patient_to_color[patient], label=patient)
        
        plt.title('UMAP Colored by Patient', fontsize=18)
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        # Place legend outside the plot
        legend = plt.legend(title="Patients", markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.setp(legend.get_title(), fontsize=12)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend

        if self.wandb_run is not None:
            self.wandb_run.log({"umap_patient_colored": wandb.Image(plt)})
        else:
            plt.savefig(os.path.join(out_dir, "umap_patient_colored.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # --- Plot 3: UMAP with Images ---
        plt.figure(figsize=(25, 20))
        ax = plt.gca()
        # Normalize embedding to be in [0,1] range for better image placement
        embedding_norm = (embedding - embedding.min(0)) / (embedding.max(0) - embedding.min(0))
        
        for i, (x, y) in enumerate(embedding_norm):
            try:
                img = Image.open(paths_list[i])
                img.thumbnail((128, 128)) # Resize for thumbnail
                im = OffsetImage(img, zoom=0.35) # Zoom can be adjusted
                ab = AnnotationBbox(im, (x, y), frameon=False, pad=0.0)
                ax.add_artist(ab)
            except FileNotFoundError:
                print(f"Warning: Image not found at {paths_list[i]}")

        ax.update_datalim(embedding_norm)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal', adjustable='box')
        plt.title('UMAP with Tile Images', fontsize=22)
        plt.xlabel('UMAP 1', fontsize=15)
        plt.ylabel('UMAP 2', fontsize=15)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

        if self.wandb_run is not None:
            self.wandb_run.log({"umap_with_images": wandb.Image(plt)})
        else:
            plt.savefig(os.path.join(out_dir, "umap_with_images.png"), dpi=300, bbox_inches="tight")
        plt.close()

        return embedding, paths_list
