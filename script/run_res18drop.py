import torch.nn as nn
import torchvision
from torchvision import models
import torch
from train import training
from model import init_res18_ciga, get_res18_dropout
import os
import torch.optim as optim

model_save_dir = "../models/res18/RUBCNL_Res18drop/"
os.makedirs(model_save_dir, exist_ok=True)

# for logging purposes
resnet = get_res18_dropout()
learning_rate = 0.0005
training(resnet=resnet,
         data_dir='../Training_Data/',
         model_save_dir=model_save_dir,
         epochs=30,
         loss_fn=nn.MSELoss(),
         optimizer=optim.AdamW([{"params": resnet.pretrained.parameters(), "lr": learning_rate},
                                {"params": resnet.gene1.parameters(), "lr": learning_rate}], weight_decay=0.005),
         learning_rate=learning_rate,
         batch_size=64,
         gene="RUBCNL")