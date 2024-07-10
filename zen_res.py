from func import MyNet, get_patient_loader
import torch
import torchvision.models as models
import zennit as zen
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.models.vgg import VGG
from torch.nn.modules.pooling import MaxPool2d, AdaptiveAvgPool2d
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from zennit.rules import Epsilon, AlphaBeta
from zennit.types import Linear
from zennit.core import Composite
from zennit.attribution import Gradient
from torchvision.models.resnet import ResNet
from torch.nn.modules.activation import ReLU
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
import os
from restructure import *

"""
different components to the other model
<class 'func.MyNet'>
<class 'torchvision.models.resnet.ResNet'>
<class 'torch.nn.modules.batchnorm.BatchNorm2d'>
<class 'torchvision.models.resnet.Bottleneck'>
<class 'torch.nn.modules.batchnorm.BatchNorm1d'>
"""

# TODO: BatchNorm2d, Bottleneck, BatchNorm1d


model = MyNet(my_pretrained_model=models.resnet50(weights="IMAGENET1K_V2"))
path = "./results_prev/results_RUBCNL/09062024_single__5e-06resnet.pt"
model.load_state_dict(torch.load(path))
model.eval()
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

# model as a sequential for the restructuring
modules = []
modules.append(model.pretrained)
for layer in model.my_new_layers:
    modules.append(layer)
for layer in model.gene1:
    modules.append(layer)

sequential = nn.Sequential(*modules)


data_dir = "./data"
patient = "/p007"
base_path = data_dir+patient+"/Preprocessed_STDataset/"
merge = pd.read_csv(base_path + "merge.csv")
merge.head()
loader = get_patient_loader(data_dir, patient)