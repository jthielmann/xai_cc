import torch.nn as nn
import torchvision
from torchvision import models
import torch
from train import training
from model import init_res18_ciga, get_res50_dropout
import os
import torch.optim as optim

model_save_dir = "../models/res50/RUBCNL_Res50_drop_freeze/"
os.makedirs(model_save_dir, exist_ok=False)

# for logging purposes
resnet = get_res50_dropout()
learning_rate = 0.0005
training(model=resnet,
         data_dir='../Training_Data/',
         model_save_dir=model_save_dir,
         epochs=40,
         loss_fn=nn.MSELoss(),
         optimizer=optim.AdamW([{"params": resnet.pretrained.parameters(), "lr": learning_rate},
                                {"params": resnet.gene1.parameters(), "lr": learning_rate}], weight_decay=0.005),
         learning_rate=learning_rate,
         batch_size=64,
         gene="RUBCNL",
         freeze_pretrained=False)