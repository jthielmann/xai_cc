import os
import lightning as L
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import pandas as pd
from umap import UMAP

from script.data_processing.lit_STDataModule import get_data_module
from script.model.lit_sae import LitSparseAutoencoder
from script.model.model_factory import get_encoder


class UMAPCallback(Callback):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.pipeline.config.get("umap"):
            return
        
        print(f"\nGenerating UMAP plots for epoch {trainer.current_epoch}...")
        self.pipeline.generate_umap(epoch=trainer.current_epoch)
        print("UMAP plots generated.")


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

        callbacks = [UMAPCallback(self)]

        self.trainer = L.Trainer(
            max_epochs=self.config["epochs"],
            logger=logger,
            accelerator="auto",
            log_every_n_steps=1,
            callbacks=callbacks,
        )

    def run(self):
        if not self.trainer:
            self.setup()

        self.trainer.fit(self.sae, train_dataloaders=self.train_loader, val_dataloaders=self.val_loader)

        if self.config.get("test_samples"):
            self.trainer.test(self.sae, datamodule=self.data_module)

    def generate_umap(self, epoch=None):
        # --- 1. Feature Extraction ---
        features_list = []
        paths_list = []
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
                
                start_idx = batch_idx * self.config['batch_size']
                end_idx = start_idx + len(imgs)
                for i in range(start_idx, end_idx):
                    paths_list.append(dataset.get_tilename(i))

        features_np = np.concatenate(features_list, axis=0)
        if features_np.ndim == 3:
            features_np = features_np.reshape(features_np.shape[0], -1)

        # --- 2. UMAP Hyperparameter Sweep ---
        
        should_generate_image_plots = self.config.get("umap_generate_image_plots", False)
        umap_sweep_params = self.config.get("umap_sweep_params", {
            "n_neighbors": [15, 30, 50],
            "min_dist": [0.1, 0.25, 0.5]
        })
        n_neighbors_list = umap_sweep_params.get("n_neighbors", [15])
        min_dist_list = umap_sweep_params.get("min_dist", [0.1])
        
        table = None
        if self.wandb_run:
            table_columns = ["Epoch", "n_neighbors", "min_dist", "Patient-Colored UMAP"]
            if should_generate_image_plots:
                table_columns.append("Image UMAP")
            table = wandb.Table(columns=table_columns)

        patient_ids = dataset.df['patient'].tolist()
        unique_patients = sorted(list(set(patient_ids)))
        cmap = plt.get_cmap('tab20', len(unique_patients))
        patient_to_color = {p: cmap(i) for i, p in enumerate(unique_patients)}
        out_dir = self.config.get("out_path") or self.config.get("sweep_dir") or self.config.get("model_dir") or "."
        os.makedirs(out_dir, exist_ok=True)

        for n_neighbors in n_neighbors_list:
            for min_dist in min_dist_list:
                umap_model = UMAP(
                    n_components=self.config.get("umap_n_components", 2),
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=42,
                    n_jobs=1
                )
                embedding = umap_model.fit_transform(features_np)

                plt.figure(figsize=(14, 10))
                for patient in unique_patients:
                    idx = [i for i, p in enumerate(patient_ids) if p == patient]
                    if not idx: continue
                    plt.scatter(embedding[idx, 0], embedding[idx, 1], s=25, color=patient_to_color[patient], label=patient)
                
                title = f'UMAP (nn={n_neighbors}, md={min_dist}, epoch={epoch})'
                plt.title(title, fontsize=18)
                plt.xlabel('UMAP 1', fontsize=12)
                plt.ylabel('UMAP 2', fontsize=12)
                legend = plt.legend(title="Patients", markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                plt.setp(legend.get_title(), fontsize=12)
                plt.tight_layout(rect=[0, 0, 0.85, 1])

                patient_plot_image = wandb.Image(plt) if self.wandb_run else None
                if not self.wandb_run:
                    filename = f"patient_colored_umap_epoch_{epoch}_nn_{n_neighbors}_md_{min_dist}.png"
                    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches="tight")
                plt.close()

                image_plot_image = None
                if should_generate_image_plots:
                    plt.figure(figsize=(25, 20))
                    ax = plt.gca()
                    embedding_norm = (embedding - embedding.min(0)) / (embedding.max(0) - embedding.min(0))
                    
                    for i, (x, y) in enumerate(embedding_norm):
                        try:
                            img = Image.open(paths_list[i])
                            img.thumbnail((128, 128))
                            im = OffsetImage(img, zoom=0.35)
                            ab = AnnotationBbox(im, (x, y), frameon=False, pad=0.0)
                            ax.add_artist(ab)
                        except FileNotFoundError:
                            print(f"Warning: Image not found at {paths_list[i]}")

                    ax.update_datalim(embedding_norm)
                    ax.set_xlim(-0.1, 1.1)
                    ax.set_ylim(-0.1, 1.1)
                    ax.set_aspect('equal', adjustable='box')
                    plt.title(f'Image UMAP (nn={n_neighbors}, md={min_dist}, epoch={epoch})', fontsize=22)
                    plt.xlabel('UMAP 1', fontsize=15)
                    plt.ylabel('UMAP 2', fontsize=15)
                    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

                    if self.wandb_run:
                        image_plot_image = wandb.Image(plt)
                    else:
                        filename = f"image_umap_epoch_{epoch}_nn_{n_neighbors}_md_{min_dist}.png"
                        plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches="tight")
                    plt.close()

                if table is not None:
                    row = [epoch, n_neighbors, min_dist, patient_plot_image]
                    if should_generate_image_plots:
                        row.append(image_plot_image)
                    table.add_data(*row)

        if table is not None:
            self.wandb_run.log({f"UMAP_Hyperparameter_Sweep_Epoch_{epoch}": table})