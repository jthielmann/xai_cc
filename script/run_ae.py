from model import get_Resnet_ae
from train import train_ae2
import torch.nn as nn
from loss_functions import SparsityLoss, CompositeLoss
import os

learning_rates = [0.01, 0.001, 0.0001, 0.0005]

model_types = ["resnet18"]
epochs = 100
training_data_dir="../NCT-CRC/"

out_dir = "../models/"
use_sparsity_loss = [True, False]


for model_type in model_types:
    for lr in learning_rates:
        for sparsity in use_sparsity_loss:
            dir_name = out_dir + "AE_" + model_type + "_"
            dir_name += "lr_" + str(lr)
            if training_data_dir.find("NCT-CRC") != -1:
                dir_name += "_NCT-CRC"
            if sparsity:
                dir_name += "_SparsityLoss"
            if os.path.exists(dir_name):
                print(dir_name + " already exists, continuing")
                continue
            try:
                os.makedirs(dir_name, exist_ok=True)
            except OSError:
                print("Creation of the directory %s failed" % dir_name)
            print(dir_name)
            model = get_Resnet_ae()
            layer_name = "encoder.encoder.5.1.conv2.1"
            if sparsity:
                losses = [nn.MSELoss(), SparsityLoss(layer_name, model)]
            else:
                losses = [nn.MSELoss()]
            loss_fn = CompositeLoss(losses)
            train_ae2(ae=model, out_dir_name=dir_name, training_data_dir=training_data_dir, criterion=loss_fn, epochs=epochs, lr=lr)
