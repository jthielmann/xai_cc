import torch.nn as nn
import torchvision
from torchvision import models
import torch
from train import training
from model import init_res18_ciga, get_res50_dropout
import os
import torch.optim as optim

os.makedirs("newStructure/RUBCNL_HLR_Res50", exist_ok=True)

# for logging purposes
resnet = get_res50_dropout()
print(type(resnet))
learning_rate = 0.0005
training(resnet=resnet,
         data_dir='../Training_Data/',
         model_save_dir="newStructure/RUBCNL_HLR_Res50/",
         epochs=30,
         loss_fn=nn.MSELoss(),
         optimizer=optim.AdamW([{"params": resnet.pretrained.parameters(), "lr": learning_rate},
                                {"params": resnet.gene1.parameters(), "lr": learning_rate}], weight_decay=0.005),
         learning_rate=learning_rate,
         batch_size=64,
         gene="RUBCNL")