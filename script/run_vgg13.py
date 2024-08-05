import torch.nn as nn
import torchvision
from torchvision import models
import torch
from train import training
from model import get_vgg13
import os
import torch.optim as optim

model_save_dir = "../models/vgg13/dropout/"
os.makedirs(model_save_dir, exist_ok=True)

# for logging purposes
resnet = get_vgg13()
learning_rate = 0.0005
training(model=resnet,
         data_dir='../Training_Data/',
         model_save_dir=model_save_dir,
         epochs=30,
         loss_fn=nn.MSELoss(),
         optimizer=optim.AdamW([{"params": resnet.pretrained.parameters(), "lr": learning_rate},
                                {"params": resnet.gene1.parameters(), "lr": learning_rate}], weight_decay=0.005),
         learning_rate=learning_rate,
         batch_size=64,
         gene="RUBCNL")