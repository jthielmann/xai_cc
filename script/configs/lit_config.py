import torch
import torchvision.models as models

debug = True
data_module_num_workers = 0
#genes = ["COL3A1", "DCN", "THY1", "ENG", "PECAM1", "TAGLN", "ACTA2", "RGS5", "SYNPO2", "CNN1", "DES", "SOX10", "S100B", "PLP1"]
genes = ["COL3A1", "DCN", "THY1", "ENG", "PECAM1", "TAGLN", "ACTA2", "RGS5", "SYNPO2", "CNN1", "DES"]
dataset = "CRC-N19"

#genes = ["RUBCNL"]
#dataset = "crc_base"
loss_fn = torch.nn.MSELoss()
error_metric_name = "MSELoss"
batch_size = 32
learning_rates = [0.01]#, 0.001, 0.0001, 0.0005]
epochs = [2]#[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
freeze_pretrained = False
encoder_type = "resnet50random"
image_size = 224

def get_encoder(encoder_type):
    if encoder_type == "dino":
        encoder = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    elif encoder_type == "resnet50random":
        encoder = models.resnet50(pretrained=False)
    else:
        raise ValueError("encoder not found")
    return encoder


def get_encoder_weights(encoder_type):
    if encoder_type == "dino":
        encoder_weights = "facebookresearch/dino:main - dino_resnet50"
    elif encoder_type == "resnet50random":
        encoder_weights = "random"
    else:
        raise ValueError("encoder not found")
    return encoder_weights


if dataset == "CRC-N19":
    mean = [0.5766, 0.3454, 0.5366]
    std = [0.2619, 0.2484, 0.2495]
    data_dir = "../data/CRC-N19/"
    if debug:
        train_samples = ["TENX92"]#,"TENX91","TENX90","TENX89","TENX70","TENX49", "ZEN49", "ZEN48", "ZEN47", "ZEN46", "ZEN45", "ZEN44"]
        val_samples = ["TENX29"]#, "ZEN43", "ZEN42", "ZEN40", "ZEN39", "ZEN38", "ZEN36"]
    else:
        train_samples = ["TENX92","TENX91","TENX90","TENX89","TENX70","TENX49", "ZEN49", "ZEN48", "ZEN47", "ZEN46", "ZEN45", "ZEN44"]
        val_samples = ["TENX29", "ZEN43", "ZEN42", "ZEN40", "ZEN39", "ZEN38", "ZEN36"]
elif dataset == "crc_base":
    mean = [0.7406, 0.5331, 0.7059]
    std = [0.1651, 0.2174, 0.1574]
    data_dir = "../data/crc_base/Training_Data/"
    if debug:
        train_samples = ["p007"]
        val_samples = ["p009"]
    else:
        train_samples = ["p007", "p014", "p016", "p020", "p025"]
        val_samples = ["p009", "p013"]
else:
    raise ValueError("dataset not found")

def get_name(epochs,
             model_name,
             learning_rate,
             encoder_type,
             error_metric_name,
             freeze_pretrained,
             dataset,
             batch_size):
    name = model_name
    name += "_ep_" + str(epochs)
    name += "_lr_" + str(learning_rate)
    name += "_" + encoder_type
    name += "_" + error_metric_name
    name += "_" + str(freeze_pretrained)
    name += "_" + dataset
    name += "_" + str(batch_size)
    for gene in genes:
        name += "_" + gene
    return name

lit_config = {
    "project": "st-xai",
    "genes": genes,
    "epochs": epochs,
    "batch_size": batch_size,
    "data_dir": data_dir,
    "freeze_pretrained": freeze_pretrained,
    "train_samples": train_samples,
    "val_samples": val_samples,
    "test_samples": None,
    "loss_fn": loss_fn,
    "error_metric_name": error_metric_name,
    "meta_data_dir_name": "meta_data",
    "encoder_type": encoder_type,
    "learning_rates": learning_rates,
    "dataset": dataset,
    "debug": debug,
    "num_workers": data_module_num_workers,
    "image_size": image_size,
    "use_transforms_in_model": False,
    "mean": mean,
    "std": std,
    "pretrained_out_dim": 1000,
}
