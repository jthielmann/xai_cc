import torch.nn as nn
import torchmetrics
import torchvision
from torchvision import models
import torch
from train import training
from model import init_res18_ciga, get_res50
import os
import torch.optim as optim

gene="MKI67"

model_save_dir = "../models/res50/" + gene + "_Res50/"
os.makedirs(model_save_dir, exist_ok=False)

# for logging purposes
resnet = get_res50()
learning_rate = 0.0005
training(model=resnet,
         data_dir='../Training_Data/',
         model_save_dir=model_save_dir,
         epochs=40,
         loss_fn=nn.L1Loss(),
         optimizer=optim.AdamW([{"params": resnet.pretrained.parameters(), "lr": learning_rate},
                                {"params": resnet.gene1.parameters(), "lr": learning_rate}], weight_decay=0.005),
         learning_rate=learning_rate,
         batch_size=64,
         gene=gene,
         freeze_pretrained=False,
         error_metric=lambda x, y: torchmetrics.functional.mean_squared_error(x, y).item(),
         error_metric_name="MSE")
