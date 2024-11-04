from model import get_Resnet_ae
from train import train_ae
import torch.nn as nn
from loss_functions import SparsityLoss
import os

learning_rates = [0.01, 0.001, 0.0001, 0.0005]

model_types = ["resnet18"]
epochs = 100
training_data_dir="../NCT-CRC/"

out_dir = "../models/"

for model_type in model_types:
    for lr in learning_rates:
        dir_name = out_dir + "AE_" + model_type + "_"
        dir_name += "lr_" + str(lr)
        if training_data_dir.find("NCT-CRC") != -1:
            dir_name += "_NCT-CRC"
        if os.path.exists(dir_name):
            print(dir_name + " already exists, continuing")
            continue
        try:
            os.makedirs(dir_name, exist_ok=True)
        except OSError:
            print("Creation of the directory %s failed" % dir_name)
        model = get_Resnet_ae()
        """for name, layer in model.named_modules():
            print(name, layer)"""

        layer_name = "encoder.encoder.5.1.conv2.1"
        train_ae(ae=model, out_dir_name=dir_name, training_data_dir=training_data_dir, criterion=[nn.MSELoss(), SparsityLoss(layer_name, model)], epochs=epochs, lr=lr)
