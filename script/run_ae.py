from script.model import get_Resnet_ae
from train import train_ae

import os


learning_rates = [0.01, 0.001, 0.0001, 0.0005]

gene_lists = [["RUBCNL"]]
model_types = ["resnet18"]
epochs = 200

for gene_list in gene_lists:
    for model_type in model_types:
        for lr in learning_rates:
            dir_name = "AE_" + model_type + "_"
            for gene in gene_list:
                dir_name += gene + "_"
            dir_name += "_lr_" + str(lr)
            if os.path.exists(dir_name):
                print(dir_name + " already exists, continuing")
                continue
            try:
                os.makedirs(dir_name, exist_ok=True)
            except OSError:
                print("Creation of the directory %s failed" % dir_name)
            model = get_Resnet_ae()

            train_ae(ae=model, dir_name=dir_name,genes=gene_list)

