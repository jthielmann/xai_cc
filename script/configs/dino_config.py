import torch
import torchvision.models as models
debug = False
data_module_num_workers = 0

data_dir = "../data/NCT-CRC-HE-100K/"
csv_name = "files.csv"
data_split = 0.8


csv_path = data_dir + csv_name
from script.data_processing.image_transforms import get_transforms
transforms = get_transforms()
bins = 1
backbone = "resnet18"

loss_fn_switch = "MSE"
error_metric_name = loss_fn_switch
batch_size = 32
learning_rates = [0.01]
epochs = [10]
encoder_type = "resnet50random"
image_size = 224
data_bins = [4]

def get_encoder(encoder_type):
    if encoder_type == "dino":
        encoder = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    elif encoder_type == "resnet50random":
        encoder = models.resnet50(pretrained=False)
    else:
        raise ValueError("encoder not found")
    return encoder


def get_pretrained_output_dim(encoder_type):
    if encoder_type == "dino":
        pretrained_out_dim = 2048
    elif encoder_type == "resnet50random":
        pretrained_out_dim = 1000
    else:
        raise ValueError("encoder not found")
    return pretrained_out_dim


dino_config = {
    "project": "st-xai-dino",
    "epochs": epochs,
    "batch_size": batch_size,
    "data_dir": data_dir,
    "test_samples": None,
    "error_metric_name": error_metric_name,
    "meta_data_dir_name": "meta_data",
    "encoder_type": encoder_type,
    "learning_rates": learning_rates,
    "debug": debug,
    "num_workers": data_module_num_workers,
    "image_size": image_size,
    "use_transforms_in_model": False,
    "mean": None,
    "std": None,
    "pretrained_out_dim": get_pretrained_output_dim(encoder_type),
    "do_hist_generation": True,
    "bins": data_bins,
    "do_profile": False,
}

if not debug:
    sweep_config = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "mse_loss"},
        "parameters": {
            "epochs": {"values": epochs},
            # "learning_rate": {"distribution": "uniform", "max": 0.1, "min": 0},
            "learning_rate": {"values": learning_rates},
            "bins": {"values": data_bins},
            #"epochs": {"values": [40]},
            #"learning_rate": {"distribution": "uniform", "max": 0.1, "min": 0},
            #"learning_rate": {"values": [0.01]},
            #"bins": {"values": [1,3,5,7,9,10]}
            #"learning_rate": {"values": [0.01]}
        },
    }
else:
    sweep_config = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "val_loss"},
        "parameters": {
            "epochs": {"values": epochs},
            #"learning_rate": {"distribution": "uniform", "max": 0.1, "min": 0},
            "learning_rate": {"values": [0.01]},
            "bins": {"values": [5]}
        },
    }

