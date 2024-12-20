import torch.optim as optim
from lightning_model import LightiningNN
import torch
import torchvision.models as models


genes = ["RUBCNL"]
error_metric_name = "MSELoss"
model = LightiningNN(genes=genes, encoder=models.resnet50(), pretrained_out_dim=1000, middel_layer_features=64,
                     error_metric_name=error_metric_name)
learning_rate = 0.001
params = []
params.append({"params": model.encoder.parameters(), "lr": learning_rate})
for gene in genes:
    params.append({"params": getattr(model, gene).parameters(), "lr": learning_rate})
optimizer = optim.AdamW(params, weight_decay=0.005)

dataset = "CRC-N19"
if dataset == "CRC-N19":
    data_dir = "../data/CRC-N19/"
    train_samples = ["TENX92","TENX91","TENX90","TENX89","TENX70","TENX49", "ZEN49", "ZEN48", "ZEN47", "ZEN46", "ZEN45", "ZEN44"]
    val_samples = ["TENX29", "ZEN43", "ZEN42", "ZEN40", "ZEN39", "ZEN38", "ZEN36"]
elif dataset == "jonas":
    data_dir = "../data/jonas/Training_Data/"
    train_samples = ["p007", "p014", "p016", "p020", "p025"]
    val_samples = ["p009", "p013"]
else:
    raise ValueError("dataset not found")

config = {
    "project": "st-xai",
    "model": model,
    "learning_rate": learning_rate,
    "genes": genes,
    "epochs": 40,
    "batch_size": 128,
    "optim": optimizer,
    #"data_dir": "../data/CRC-N19/",
    "data_dir": data_dir,
    "output_dir": "../models/ligthning_test/",
    "freeze_pretrained": False,
    "train_samples": train_samples,
    "val_samples": val_samples,
    "test_samples": None,
    "loss_fn": torch.nn.MSELoss(),
    "error_metric_name":error_metric_name,
    "meta_data_dir_name": "meta_data"
}
