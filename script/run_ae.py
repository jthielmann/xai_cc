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
gene_lists = [["FLT1", "VWF", "PECAM1", "PLVAP", "DES"], ["RUBCNL"], ["MKI67"]]
model_types = ["resnet18d", "resnet50d"]
epochs = 40

configs = []
configs.append(())

for config in configs:
    dir_name_base = "/" + genes[0]
    for gene in genes[1:]:
        dir_name_base += "_" + gene


    random_weights_bool = [True, False]
    dropout_bool = [True, False]
    freeze_bool = [True, False]


    training_ae(model=model,
               data_dir='../Training_Data/',
               model_save_dir=dir_name,
               epochs=epochs,
               loss_fn=nn.MSELoss(),
               optimizer=optim.AdamW(params, weight_decay=0.005),
               learning_rate=learning_rate,
               batch_size=128,
               )
