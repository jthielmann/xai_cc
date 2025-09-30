
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from script.data_processing.lit_STDataModule import get_data_module
from script.model.lit_sae import LitSparseAutoencoder
from script.model.model_factory import get_encoder
import wandb

class SAETrainerPipeline:
    def __init__(self, config: dict, run: wandb.sdk.wandb_run.Run):
        self.config = config
        self.run = run
        self.sae = None
        self.data_module = None
        self.trainer = None
        self.encoder = None

    def setup(self):
        """Initializes all the components for training."""
        # 1. Initialize DataModule
        self.data_module = get_data_module(self.config)

        # 2. Initialize Model
        self.encoder = get_encoder(self.config.get("encoder_type"))
        self.sae = LitSparseAutoencoder(self.config)

        # 3. Initialize Logger
        logger = None
        if self.config.get("log_to_wandb"):
            logger = WandbLogger(project=self.config["project"], name=self.config["name"])

        # 4. Initialize Trainer
        self.trainer = L.Trainer(
            max_epochs=self.config["epochs"],
            logger=logger,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=1,
        )

    def run(self):
        """Runs the training and testing loops."""
        if not self.trainer:
            self.setup()

        # Run Training
        # TODO: needs a fix so that we train on both encoder feeds into sae, encoder must be frozen
        self.trainer.fit(self.sae, datamodule=self.data_module)

        # Run Testing (optional)
        if self.config.get("test_samples"):
            self.trainer.test(self.sae, datamodule=self.data_module)

        if self.config.get("umap"):
            self.trainer.generate_umap()

    # TODO: generates a umap of the output of the encoder + sae, logs to self.run
    def generate_umap(self):
        pass


