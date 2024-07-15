import torch.nn as nn
import torchvision
from torchvision import models
import torch
from func import get_model, get_model2, training




# for logging purposes
training(resnet=Res18(ciga),
         data_dir='../Training_Data/',
         model_save_dir="./RUBCNL_HLR_Res18/",
         epochs=30,
         loss_fn=nn.MSELoss(),
         learning_mode="HLR",
         batch_size=64,
         gene="RUBCNL")