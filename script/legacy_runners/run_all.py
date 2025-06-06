import torch.nn as nn

from script.train.train import training_multi

import torchmetrics

from script.model.model import general_model
import os
import torch.optim as optim
from script.model.loss_functions import CompositeLoss
import pandas as pd

learning_rates = [0.001, 0.01, 0.0001]
output_dir = "../../models/"
meta_data_dir = "/meta_data/"
#data_dir = "../new_Training_Data/N20/"
data_dir = '../../data/CRC-N19/'

#gene_lists = [["RUBCNL"]]
#, "MYH11"
#gene_lists = [["COL3A1","DCN","THY1"], ["ENG", "PECAM1"], ["TAGLN", "ACTA2", "RGS5"], ["TAGLN", "ACTA2", "SYNPO2", "CNN1", "DES"], ["SOX10", "S100B", "PLP1"]]
gene_lists = [["COL3A1","DCN","THY1"]]
check_if_gene_in_dataset = True
if check_if_gene_in_dataset:
    patients_dir = data_dir
    patients = [patient for patient in os.listdir(patients_dir) if os.path.isdir(patients_dir + "/" + patient)]
    df = pd.read_csv(data_dir + patients[0]+ "/" + meta_data_dir + "/gene_data.csv")
    print(df.columns)
    ids_to_remove = []
    for gene_list in gene_lists:
        for gene in gene_list:
            if gene not in df.columns:
                print(gene, "not in dataset")
                exit(1)
            else:
                print(gene, "in dataset")
                continue
    ids_to_remove.sort(reverse=True)
    for i in ids_to_remove:
        gene_lists.remove(gene_lists[i])
print(gene_lists)

model_types = ["resnet18", "resnet50", "resnet50d", "resnet18d", "resnet50dino"]

epochs = 40
random_weights_bool = [True, False]
freeze_bool = [False]

use_default_samples = False
samples = [["TENX92","TENX91","TENX90","TENX89","TENX70","TENX49", "ZEN49", "ZEN48", "ZEN47", "ZEN46", "ZEN45", "ZEN44"],
           ["TENX29", "ZEN43", "ZEN42", "ZEN40", "ZEN39", "ZEN38", "ZEN36"]]

appendix = "_CRC-N19"

for genes in gene_lists:
    dir_name_base = "/" + genes[0]
    for gene in genes[1:]:
        dir_name_base += "_" + gene

    for model_type in model_types:
        for learning_rate in learning_rates:
            dir_name = output_dir + model_type + dir_name_base
            for use_random_weights in random_weights_bool:
                for do_freeze_pretrained in freeze_bool:
                    dir_name = output_dir + model_type + dir_name_base
                    if use_random_weights:
                        dir_name += "_random"
                    if do_freeze_pretrained:
                        dir_name += "_freeze"
                    dir_name += "_lr_" + str(learning_rate)
                    if appendix:
                        dir_name += appendix
                    max_file_length = 255
                    dir_name += "/"
                    if len(dir_name) > max_file_length:
                        print(dir_name, "is longer than max_file_length", max_file_length)
                    if os.path.exists(dir_name):
                        print(dir_name, "already exists, continue..")
                        continue
                    os.makedirs(dir_name, exist_ok=True)

                    model = general_model(model_type, genes, random_weights=use_random_weights, dropout=False, pretrained_out_dim=1000)
                    print(dir_name)
                    print(len(dir_name))

                    params = []
                    params.append({"params": model.pretrained.parameters(), "lr": learning_rate})
                    for gene in genes:
                        params.append({"params": getattr(model, gene).parameters(), "lr": learning_rate})
                    losses = [nn.MSELoss()]
                    loss_fn = CompositeLoss(losses)
                    training_multi(model=model,
                                   data_dir=data_dir,
                                   model_save_dir=dir_name,
                                   epochs=epochs,
                                   loss_fn=loss_fn,
                                   optimizer=optim.AdamW(params, weight_decay=0.005),
                                   learning_rate=learning_rate,
                                   batch_size=128,
                                   genes=genes,
                                   freeze_pretrained=do_freeze_pretrained,
                                   error_metric=lambda x, y: torchmetrics.functional.mean_squared_error(x, y).item(),
                                   error_metric_name="MSE",
                                   meta_data_dir_name=meta_data_dir,
                                   use_default_samples=use_default_samples,
                                   samples=samples)
