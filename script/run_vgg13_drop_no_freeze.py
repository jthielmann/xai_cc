import torch.nn as nn
import torchvision
from torchvision import models
import torch
from train import training
from model import get_vgg13_dropout, get_vggs_and_path, get_resnets_and_path
import os
import torch.optim as optim
import random
model_save_dir = "../models/vgg13/dropout/"
os.makedirs(model_save_dir, exist_ok=True)
vgg = get_vgg13_dropout()
learning_rate = 0.00005
freeze_pretrained = False
weight_decay = 0.005
if freeze_pretrained:
    optimizer = optim.SGD([{"params": vgg.pretrained.parameters(), "lr": learning_rate},
                            {"params": vgg.gene1.parameters(), "lr": learning_rate}], weight_decay=weight_decay)
else:
    optimizer = optim.SGD([{"params": vgg.gene1.parameters(), "lr": learning_rate}], weight_decay=weight_decay)

print("base vgg model, model.gene1.eval(), freeze pretrained, no decay")


training(model=vgg,
         data_dir='../Training_Data/',
         model_save_dir=model_save_dir,
         epochs=40,
         loss_fn=nn.MSELoss(),
         optimizer=optimizer,
         learning_rate=learning_rate,
         batch_size=64,
         gene="RUBCNL",
         freeze_pretrained=freeze_pretrained)

