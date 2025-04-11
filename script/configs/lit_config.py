import torch
import torchvision.models as models
import torch.nn as nn
debug = False
data_module_num_workers = 0

genes = ["RUBCNL"]
dataset = "crc_base"
loss_fn_switch = "MSE"
error_metric_name = loss_fn_switch
batch_size = 32
learning_rates = [0.01, 0.1, 0.001, 0.0001] if not debug else [0.01]
epochs = [40 if not debug else 3]
freeze_pretrained = True
encoder_type = "dino"
image_size = 224
data_bins = [1, 3, 5, 7, 9, 10]
gene_data_csv_filename = "gene_data_ranknorm.csv"#"gene_data_raw.csv"#"gene_data_ranknorm.csv"#"gene_data.csv"#"gene_data_raw.csv" #"gene_data_log1p.csv"

def get_encoder(encoder_type):
    if encoder_type == "dino":
        encoder = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    elif encoder_type == "resnet50random":
        encoder = models.resnet50(pretrained=False)
    elif encoder_type == "resnet50imagenet":
        encoder = models.resnet50(weights="IMAGENET1K_V2")
    else:
        raise ValueError("encoder not found")
    return encoder


def get_pretrained_output_dim(encoder_type):
    if encoder_type == "dino":
        pretrained_out_dim = 2048
    elif encoder_type == "resnet50random" or encoder_type == "resnet50imagenet":
        pretrained_out_dim = 1000
    else:
        raise ValueError("encoder not found")
    return pretrained_out_dim


if dataset == "CRC_N19":
    #mean = [0.0, 0.0, 0.0] # mit val
    #std = [1, 1, 1] # mit val
    #mean = [0.7406, 0.5331, 0.7059] # original base
    #std = [0.1651, 0.2174, 0.1574] # original base
    mean = [0.0555, 0.1002, 0.00617] # ohne val
    std = [0.991, 0.9826, 0.9967] # ohne val
    data_dir = "../data/CRC-N19/"
    if debug:
        train_samples = ["TENX92"]#,"TENX91","TENX90","TENX89","TENX70","TENX49", "ZEN49", "ZEN48", "ZEN47", "ZEN46", "ZEN45", "ZEN44"]
        val_samples = ["TENX29"]#, "ZEN43", "ZEN42", "ZEN40", "ZEN39", "ZEN38", "ZEN36"]
    else:
        train_samples = ["TENX92","TENX91","TENX90","TENX89","TENX70","TENX49", "ZEN49", "ZEN48", "ZEN47", "ZEN46", "ZEN45", "ZEN44"]
        val_samples = ["TENX29", "ZEN43", "ZEN42", "ZEN40", "ZEN39", "ZEN38", "ZEN36"]
elif dataset == "CRC-N19_2": # different preprocessing
    mean = [0.5405, 0.2749, 0.5476]  # ohne val
    std = [0.2619, 0.2484, 0.2495]  # ohne val
    data_dir = "../data/CRC-N19/"
    if debug:
        train_samples = [
            "TENX92"]  # ,"TENX91","TENX90","TENX89","TENX70","TENX49", "ZEN49", "ZEN48", "ZEN47", "ZEN46", "ZEN45", "ZEN44"]
        val_samples = ["TENX29"]  # , "ZEN43", "ZEN42", "ZEN40", "ZEN39", "ZEN38", "ZEN36"]
    else:
        train_samples = ["TENX92", "TENX91", "TENX90", "TENX89", "TENX70", "TENX49", "ZEN49", "ZEN48", "ZEN47",
                         "ZEN46", "ZEN45", "ZEN44"]
        val_samples = ["TENX29", "ZEN43", "ZEN42", "ZEN40", "ZEN39", "ZEN38", "ZEN36"]
elif dataset == "crc_base":
    #mean = [0.7406, 0.5331, 0.7059] # original base
    #std = [0.1651, 0.2174, 0.1574] # original base
    #mean = [-0.4683, -0.1412, -0.4501]
    #std = [1.7697, 1.3199, 1.7859] # only train
    mean = [0.331, 0.632, 0.3946] # calculated only for the training data
    std = [1.1156, 1.1552, 1.1266] # calculated only for the training data

    data_dir = "../data/crc_base/Training_Data/"
    if debug:
        train_samples = ["p007"]
        val_samples = ["p009"]
    else:
        train_samples = ["p007", "p014", "p016", "p020", "p025"]
        val_samples = ["p009", "p013"]
elif dataset == "pseudospot":
    mean = [0.331, 0.632, 0.3946] # calculated only for the training data
    std = [1.1156, 1.1552, 1.1266] # calculated only for the training data

    data_dir = "../data/pseudospot/"
    if debug:
        train_samples = ["p007"]
        val_samples = ["p009"]
    else:
        train_samples = ["p007", "p014", "p016", "p020", "p025"]
        val_samples = ["p009", "p013"]
else:
    raise ValueError("dataset not found")

if loss_fn_switch == "MSE":
    loss_fn = torch.nn.MSELoss()
elif loss_fn_switch == "Weighted MSE":
    def get_mse_weights(dataset):
        if dataset == "CRC-N19":
            mse_weights = [4.28082192e-04, 4.56933973e-05, 3.49601454e-05, 3.11390671e-05, 3.15159155e-05, 3.25563224e-05,
                           7.15819613e-05, 9.69461949e-05, 1.11656990e-04, 1.13533152e-04, 1.17274540e-04, 1.70270730e-04,
                           2.07555002e-04, 2.73448182e-04, 4.90918017e-04, 8.84955752e-04, 2.01612903e-03, 4.76190476e-03,
                           2.50000000e-02, 2.50000000e-02, 2.56410256e-02, 2.63157895e-02, 2.85714286e-02, 6.25000000e-02,
                           1, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00]
        elif dataset == "crc_base":
            counts_after_smoothing = [6478,9043,9994,9530,6989,6769,4580,5883,5732,5595,5293,5827,4186,4467,4476,3725,3581,3262,2488,1912,1333,861,563,332,191,118,66,38,15,5,1]
            sum_counts = sum(counts_after_smoothing)
            mse_weights = [sum_counts / i if i != 0 else 1 for i in counts_after_smoothing]
        elif dataset == "CRC-N19_2":
            counts_after_smoothing = [2336,21885,28604,32114,31730,30716,13970,10315,8956,8808,8527,5873,4818,3657,2037,1130,496,210,40,40,39,38,35,16,0,1,1,1,1,1,1,]
            sum_counts = sum(counts_after_smoothing)
            mse_weights = [sum_counts / i if i != 0 else 1 for i in counts_after_smoothing]
        else:
            raise ValueError("dataset not implemented for get_mse_weights")
        return torch.tensor(mse_weights)
    print("using weighted mse loss")


    class weighted_mse_loss(nn.Module):
        def __init__(self, weights):
            super(weighted_mse_loss, self).__init__()
            self.weights = weights

        def forward(self, inputs, targets):
            loss = (inputs - targets) ** 2
            loss *= self.weights.expand_as(loss)
            loss = torch.mean(loss)
            return loss

    loss_fn = weighted_mse_loss(get_mse_weights(dataset))
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
    "pretrained_out_dim": get_pretrained_output_dim(encoder_type),
    "do_hist_generation": True,
    "bins": data_bins,
    "do_profile": False,
    "gene_data_filename_train":gene_data_csv_filename,
    "gene_data_filename_val":gene_data_csv_filename,
    "gene_data_filename_test":gene_data_csv_filename
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
        "metric": {"goal": "minimize", "name": "mse_loss"},
        "parameters": {
            "epochs": {"values": epochs},
            #"learning_rate": {"distribution": "uniform", "max": 0.1, "min": 0},
            "learning_rate": {"values": [0.01]},
            "bins": {"values": [5]}
        },
    }

