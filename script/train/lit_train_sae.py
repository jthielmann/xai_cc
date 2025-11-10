import os
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
import wandb
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import pandas as pd
from umap import UMAP

from script.data_processing.lit_STDataModule import get_data_module
from script.model.lit_sae import LitSparseAutoencoder
from script.model.preproc import LinearPCATransform
from sklearn.decomposition import PCA


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


class WandbMetricCallback(Callback):
    """Forward selected metrics to an existing W&B run handle.

    This is useful when Lightning's WandbLogger is disabled, but we still want
    to log key metrics (e.g., in nested pipelines).
    """
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    def on_validation_epoch_end(self, trainer, pl_module):
        run = getattr(self.pipeline, "wandb_run", None)
        if run is None:
            return
        metrics = trainer.callback_metrics
        payload = {"epoch": trainer.current_epoch}
        val_cos = metrics.get("val_cosine_sae")
        if val_cos is not None:
            v = val_cos.detach().cpu().item() if hasattr(val_cos, "detach") else val_cos
            if torch.is_tensor(v):
                v = v.item()
            if not isinstance(v, (int, float)):
                raise TypeError(f"val_cosine_sae type unsupported: {type(v)}")
            payload["val_cosine_sae"] = float(v)
        val_mse = metrics.get("val_MSELoss_sae")
        if val_mse is not None:
            v = val_mse.detach().cpu().item() if hasattr(val_mse, "detach") else val_mse
            if torch.is_tensor(v):
                v = v.item()
            if not isinstance(v, (int, float)):
                raise TypeError(f"val_MSELoss_sae type unsupported: {type(v)}")
            payload["val_MSELoss_sae"] = float(v)
        if len(payload) > 1:
            run.log(payload)


class SAETrainerPipeline:
    def __init__(self, config: dict, run: wandb.sdk.wandb_run.Run, *, encoder: nn.Module, gene_head=None):
        self.config = config
        self.wandb_run = run
        self.encoder = encoder
        if self.encoder is None:
            raise ValueError("SAETrainerPipeline requires a non-None encoder")
        self.gene_head = gene_head
        self.sae = None
        self.data_module = get_data_module(self.config)
        self.data_module.setup("fit")
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader   = self.data_module.val_dataloader()
        self.test         = self.data_module.test_dataloader()
        self.trainer = None

    def setup(self):

        # Infer d_in from encoder output (flatten if needed) if not specified
        if "d_in" not in self.config:
            sample_batch = next(iter(self.train_loader))
            # The batch is either a tensor or a list/tuple.
            if isinstance(sample_batch, (list, tuple)):
                sample_input = sample_batch[0]
            else:
                sample_input = sample_batch
            
            with torch.no_grad():
                encoder_output = self.encoder(sample_input)
            
            if encoder_output.ndim == 2:
                d_in = int(encoder_output.shape[-1])
            else:
                d_in = int(np.prod(encoder_output.shape[1:]))
            self.config['d_in'] = d_in

        preproc = None
        use_pca = bool(self.config.get("pca_enable", False) or self.config.get("use_pca", False) or (self.config.get("pca_k") is not None))
        if use_pca:
            k = int(self.config.get("pca_k", self.config['d_in']))
            if not (1 <= k <= int(self.config['d_in'])):
                raise ValueError(f"pca_k must be in [1, d_in], got k={k}, d_in={self.config['d_in']}")
            whiten = bool(self.config.get("pca_whiten", True))
            fit_cap = self.config.get("pca_fit_max_samples")
            feats = []
            seen = 0
            dev = next(self.encoder.parameters()).device
            self.encoder.eval()
            with torch.no_grad():
                for batch in self.train_loader:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    x = x.to(dev)
                    z = self.encoder(x)
                    if z.ndim == 4:
                        z = z.mean(dim=(2, 3))
                    elif z.ndim == 3:
                        z = z.mean(dim=1)
                    z = z.view(z.size(0), -1)
                    feats.append(z.detach().cpu().numpy().astype(np.float32))
                    seen += z.size(0)
                    if fit_cap is not None and seen >= int(fit_cap):
                        break
            if not feats:
                raise RuntimeError("no features collected for PCA fit")
            X = np.concatenate(feats, axis=0)
            if fit_cap is not None:
                X = X[: int(fit_cap)]
            p = PCA(n_components=k, whiten=False, svd_solver='auto', random_state=42)
            p.fit(X)
            mean = torch.from_numpy(p.mean_.astype(np.float32))
            comps = torch.from_numpy(p.components_.astype(np.float32))
            var = torch.from_numpy(p.explained_variance_.astype(np.float32))
            preproc = LinearPCATransform(mean, comps, var, whiten=whiten)
            self.config['d_in'] = int(k)

        self.sae = LitSparseAutoencoder(self.config, encoder=self.encoder, preproc=preproc)

        logger = None
        if self.config.get("log_to_wandb"):
            logger = WandbLogger(project=self.config["project"], name=self.config["name"], log_model=False)

        callbacks: list[Callback] = [UMAPCallback(self)]
        if logger is None and self.wandb_run is not None:
            callbacks.append(WandbMetricCallback(self))
        
        if self.config.get("model_dir"):
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.config.get("model_dir"),
                filename='{epoch}-{val_cosine_sae:.4f}',
                save_top_k=1,
                monitor='val_cosine_sae',
                mode='max'
            )
            callbacks.append(checkpoint_callback)

        self.trainer = L.Trainer(
            max_epochs=self.config["epochs"],
            logger=logger,
            accelerator="auto",
            log_every_n_steps=1,
            callbacks=callbacks,
        )

    def run(self) -> Tuple[nn.Module, nn.Module]:
        if not self.trainer:
            self.setup()

        self.trainer.fit(self.sae, train_dataloaders=self.train_loader, val_dataloaders=self.val_loader)

        if self.config.get("test_samples"):
            self.trainer.test(self.sae, datamodule=self.data_module)
        # Return the encoder and the trained SAE module (not the Lightning wrapper)
        return self.encoder, self.sae.sae

    def generate_umap(self, epoch=None):
        # --- 1. Feature Extraction ---
        features_list = []
        paths_list = []
        dataset = self.data_module.val_dataset

        max_samples = self.config.get("umap_max_samples")
        num_samples = 0

        device = getattr(self.trainer, "device", None)
        if device is None:
            device = getattr(getattr(self.trainer, "strategy", None), "root_device", None) or (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            )
        self.sae.to(device)
        with torch.no_grad():
            for batch_idx, imgs in enumerate(self.val_loader):
                if max_samples is not None and num_samples >= max_samples:
                    break
                imgs = imgs.to(device)
                feats = self.encoder(imgs)
                _ = self.sae(feats)
                codes = self.sae.sae.last_sparse
                features_list.append(codes.cpu().numpy())

                start_idx = batch_idx * self.config['batch_size']
                end_idx = start_idx + len(imgs)
                for i in range(start_idx, end_idx):
                    paths_list.append(dataset.get_tilename(i))
                num_samples += len(imgs)

        features_np = np.concatenate(features_list, axis=0)
        if max_samples is not None and features_np.shape[0] > max_samples:
            features_np = features_np[:max_samples]
            paths_list = paths_list[:max_samples]

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
        if max_samples is not None:
            patient_ids = patient_ids[:max_samples]
            
        unique_patients = sorted(list(set(patient_ids)))
        cmap = plt.get_cmap('tab20', len(unique_patients))
        patient_to_color = {p: cmap(i) for i, p in enumerate(unique_patients)}
        out_dir = self.config.get("out_path") or self.config.get("sweep_dir") or self.config.get("model_dir") or "."
        if self.config.get("umap") and out_dir == ".":
            raise ValueError("UMAP enabled but no out_path/sweep_dir/model_dir set")
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
                        pth = paths_list[i]
                        if os.path.isfile(pth):
                            img = Image.open(pth)
                            img.thumbnail((128, 128))
                            im = OffsetImage(img, zoom=0.35)
                            ab = AnnotationBbox(im, (x, y), frameon=False, pad=0.0)
                            ax.add_artist(ab)
                        else:
                            print(f"Warning: Image not found at {pth}")

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
