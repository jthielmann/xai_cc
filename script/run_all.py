import torch.nn as nn
from train import training, training_multi

import torchmetrics

from model import general_model
import os
import torch.optim as optim



learning_rate = 0.0005

#gene="MKI67"
#gene="RUBCNL"
#gene_lists = [["VWF"]]
gene_lists = [["RUBCNL"]]
model_types = ["resnet18"]
epochs = 40

for genes in gene_lists:
    dir_name_base = "/" + genes[0]
    for gene in genes[1:]:
        dir_name_base += "_" + gene


    random_weights_bool = [True]
    dropout_bool = [True]
    dropout_values = [round(0.1 * x, 1) for x in range(10)]
    freeze_bool = [False]

    for model_type in model_types:
        dir_name = "../models/" + model_type + dir_name_base


        for use_random_weights in random_weights_bool:
            for use_dropout in dropout_bool:
                for do_freeze_pretrained in freeze_bool:
                    for dropout_value in dropout_values:
                        dir_name = "../models/" + model_type + dir_name_base
                        if use_random_weights:
                            dir_name += "_random"
                        if do_freeze_pretrained:
                            dir_name += "_freeze"
                        if use_dropout:
                            dir_name += "_dropout_" + str(dropout_value)
                        max_file_length = 255

                        dir_name += "/"
                        if len(dir_name) > max_file_length:
                            print(dir_name, "is longer than max_file_length", max_file_length)
                        if os.path.exists(dir_name):
                            print(dir_name, "already exists, continue..")
                            continue
                        os.makedirs(dir_name, exist_ok=True)

                        model = general_model(model_type, genes, use_random_weights, use_dropout, dropout_value, 1000)
                        print(dir_name)
                        print(len(dir_name))

                        params = []
                        params.append({"params": model.pretrained.parameters(), "lr": learning_rate})
                        for gene in genes:
                            params.append({"params": getattr(model, gene).parameters(), "lr": learning_rate})
                        training_multi(model=model,
                                       data_dir='../Training_Data/',
                                       model_save_dir=dir_name,
                                       epochs=epochs,
                                       loss_fn=nn.MSELoss(),
                                       optimizer=optim.AdamW(params, weight_decay=0.005),
                                       learning_rate=learning_rate,
                                       batch_size=128,
                                       genes=genes,
                                       freeze_pretrained=do_freeze_pretrained,
                                       error_metric=lambda x, y: torchmetrics.functional.mean_squared_error(x, y).item(),
                                       error_metric_name="MSE")
