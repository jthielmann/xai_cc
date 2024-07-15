import os
import cv2
import random

import numpy as np
import torch.nn as nn
import torch.nn.functional
import torch
import torchvision
from torchvision import transforms, models

# debug
from inspect import currentframe, getframeinfo

DEFAULT_RANDOM_SEED = 42
print(getframeinfo(currentframe()).lineno)


def seed_basic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def seed_torch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_everything(seed=DEFAULT_RANDOM_SEED):
    seed_basic(seed)
    seed_torch(seed)


seed_everything(seed=DEFAULT_RANDOM_SEED)


class MyNet(nn.Module):
    def __init__(self, my_pretrained_model):
        super(MyNet, self).__init__()
        self.pretrained = my_pretrained_model
        self.my_new_layers = nn.Sequential(nn.Linear(1000, 200),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(200),
                                           nn.Dropout(0.3),
                                           nn.Linear(200, 10),  #choose x in nn.Linear(20,x) depending on n_classes
                                           nn.ReLU(),
                                           nn.Dropout(0.3))

        self.gene1 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

    def forward(self, x):
        x = self.pretrained(x)
        x = self.my_new_layers(x)
        x = self.gene1(x)
        return x.squeeze()

    # is fundamentally broken, just a quick workaround as the restructuring expected a sequential to access the upper layers
    def __getitem__(self, index):
        return self.gene1[index]



class MyNet2(nn.Module):
    def __init__(self):
        super(MyNet2, self).__init__()
        self.pretrained = models.resnet50(weights="IMAGENET1K_V2")
        self.gene1 = nn.Sequential(nn.Linear(1000, 200), nn.ReLU(), nn.Linear(200, 1))

    def forward(self, x):
        return self.gene1(self.pretrained(x))


class Res18(nn.Module):
    def __init__(self):
        super(Res18, self).__init__()
        self.pretrained = models.resnet18()
        self.gene1 = nn.Sequential(nn.Linear(512, 200), nn.ReLU(), nn.Linear(200, 1))

    def forward(self, x):
        return self.pretrained(self.gene1(x))


def load_res18_ciga(path):
    def load_model_weights(model, weights):
        model_dict = model.state_dict()
        weights = {k: v for k, v in weights.items() if k in model_dict}
        if weights == {}:
            print('No weight could be loaded..')
        model_dict.update(weights)
        model.load_state_dict(model_dict)

        return model

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = torchvision.models.__dict__['resnet18'](pretrained=False)
    state = torch.load(path, map_location=device)

    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

    ciga = load_model_weights(model, state_dict)
    ciga.fc = torch.nn.Sequential()
    return ciga


def get_res18(path=None):
    res18 = Res18()
    if path:
        print(res18.load_state_dict(torch.load(path, map_location=torch.device('cpu'))))
    return res18


def get_res50(path=None):
    res50 = MyNet2()
    if path:
        print(res50.load_state_dict(torch.load(path, map_location=torch.device('cpu'))))
    return res50



