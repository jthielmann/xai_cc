import torch.nn as nn
import torchvision
from torchvision import models
import torch
from train import training
from model import get_vgg13, get_vggs_and_path, get_resnets_and_path
import os
import torch.optim as optim
import random
model_save_dir = "../models/vgg13/dropout_test1/"
os.makedirs(model_save_dir, exist_ok=True)
vgg = get_vgg13()
learning_rate = 0.05
freeze_pretrained = True
weight_decay = 0.005
if freeze_pretrained:
    optimizer = optim.SGD([{"params": vgg.pretrained.parameters(), "lr": learning_rate},
                            {"params": vgg.gene1.parameters(), "lr": learning_rate}], weight_decay=weight_decay)
else:
    optimizer = optim.SGD([{"params": vgg.gene1.parameters(), "lr": learning_rate}], weight_decay=weight_decay)

training(model=vgg,
         data_dir='../Training_Data/',
         model_save_dir=model_save_dir,
         epochs=100,
         loss_fn=nn.MSELoss(),
         optimizer=optimizer,
         learning_rate=learning_rate,
         batch_size=64,
         gene="RUBCNL",
         freeze_pretrained=freeze_pretrained)