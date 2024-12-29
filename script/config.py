import torch.optim as optim
from lightning_model import LightiningNN
import torch
import os

#genes = ["RUBCNL"]
genes = ["COL3A1", "DCN", "THY1", "ENG", "PECAM1", "TAGLN", "ACTA2", "RGS5", "SYNPO2", "CNN1", "DES", "SOX10", "S100B", "PLP1"]
error_metric_name = "MSELoss"
out_dir = "../models/ligthning_test/"
if os.path.exists(out_dir):
    print("out_dir already exists")
    #exit(1)
else:
    os.makedirs(out_dir)
freeze_pretrained = False
model = LightiningNN(genes=genes, encoder=torch.hub.load('facebookresearch/dino:main', 'dino_resnet50'), pretrained_out_dim=2048, middel_layer_features=64,
                     error_metric_name=error_metric_name, out_path=out_dir, freeze_pretrained=freeze_pretrained)

learning_rate = 0.001
params = []
params.append({"params": model.encoder.parameters(), "lr": learning_rate / 100})
for gene in genes:
    params.append({"params": getattr(model, gene).parameters(), "lr": learning_rate})
optimizer = optim.AdamW(params, weight_decay=0.005)

dataset = "CRC-N19"
if dataset == "CRC-N19":
    data_dir = "../data/CRC-N19/"
    train_samples = ["TENX92"]#,"TENX91","TENX90","TENX89","TENX70","TENX49", "ZEN49", "ZEN48", "ZEN47", "ZEN46", "ZEN45", "ZEN44"]
    val_samples = ["TENX29"]#, "ZEN43", "ZEN42", "ZEN40", "ZEN39", "ZEN38", "ZEN36"]
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
    "epochs": 1,
    "batch_size": 256,
    "optim": optimizer,
    "data_dir": data_dir,
    "output_dir": out_dir,
    "freeze_pretrained": freeze_pretrained,
    "train_samples": train_samples,
    "val_samples": val_samples,
    "test_samples": None,
    "loss_fn": torch.nn.MSELoss(),
    "error_metric_name": error_metric_name,
    "meta_data_dir_name": "meta_data",
    "encoder_weights": "dino"
}
