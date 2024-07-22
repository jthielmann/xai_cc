import torch.nn as nn
import torchvision
from torchvision import models
import torch
from train import training
from model import init_res18_ciga


# for logging purposes
training(resnet=init_res18_ciga(),
         data_dir='../Training_Data/',
         model_save_dir="./RUBCNL_HLR_Res18/",
         epochs=30,
         loss_fn=nn.MSELoss(),
         learning_mode="HLR",
         batch_size=64,
         gene="RUBCNL")