import torch.nn as nn
import torch.nn.functional
import torch
import torchvision
from torchvision import models


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.pretrained = models.resnet50(weights="IMAGENET1K_V2")
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


def get_MyNet(path=None):
    model = MyNet()
    if path:
        print(model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))))
    return model


class MyNet2(nn.Module):
    def __init__(self):
        super(MyNet2, self).__init__()
        self.pretrained = models.resnet50(weights="IMAGENET1K_V2")
        self.gene1 = nn.Sequential(nn.Linear(1000, 200), nn.ReLU(), nn.Linear(200, 1))

    def forward(self, x):
        return self.gene1(self.pretrained(x))


def get_res50(path=None):
    res50 = MyNet2()
    if path:
        print(res50.load_state_dict(torch.load(path, map_location=torch.device('cpu'))))
    return res50


class Res50Dropout(nn.Module):
    def __init__(self):
        super(Res50Dropout, self).__init__()
        self.pretrained = models.resnet50(weights="IMAGENET1K_V2")
        self.gene1 = nn.Sequential(nn.Linear(1000, 200), nn.Dropout(),
                                   nn.ReLU(),
                                   nn.Linear(200, 1), nn.Dropout())

    def forward(self, x):
        return self.gene1(self.pretrained(x))


def get_res50_dropout(path=None):
    res50 = Res50Dropout()
    if path:
        print(res50.load_state_dict(torch.load(path, map_location=torch.device('cpu'))))
    return res50


# old model from jonas
# path = "../../models/res50/10082023_RUBCNL_ST_absolute_single_64_NLR_loss_resnet50.pt"
class MyNet3(nn.Module):
    def __init__(self, my_pretrained_model):
        super(MyNet3, self).__init__()
        self.pretrained = my_pretrained_model

        self.my_new_layers = nn.Sequential(nn.Linear(1000, 200),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(200),
                                           nn.Dropout(0.3),
                                           nn.Linear(200, 1))

    def forward(self, x):
        x = self.pretrained(x)
        output = self.my_new_layers(x)
        return output


class Res18(nn.Module):
    def __init__(self, pretrained=models.resnet18()):
        super(Res18, self).__init__()
        self.pretrained = pretrained
        self.gene1 = nn.Sequential(nn.Linear(512, 200), nn.ReLU(), nn.Linear(200, 1))

    def forward(self, x):
        return self.gene1(self.pretrained(x))


def load_res18_ciga(path):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = torchvision.models.__dict__['resnet18'](pretrained=False)
    state = torch.load(path, map_location=device)

    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

    model_dict = model.state_dict()
    weights = {k: v for k, v in state_dict.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    model.fc = torch.nn.Sequential()
    return model


def init_res18_ciga(path="../models/res18/tenpercent_resnet18.ckpt"):
    ciga = load_res18_ciga(path)
    return Res18(ciga)


def get_res18_ciga(path):
    ciga = models.resnet18()
    ciga.fc = torch.nn.Sequential()
    model = Res18(ciga)
    print(model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))))
    return model


def get_res18(path=None):
    res18 = Res18()
    if path:
        print(res18.load_state_dict(torch.load(path, map_location=torch.device('cpu'))))
    return res18


class Res18Dropout(nn.Module):
    def __init__(self, ciga):
        super(Res18Dropout, self).__init__()
        self.pretrained = ciga
        self.gene1 = nn.Sequential(nn.Linear(512, 200), nn.Dropout(), nn.ReLU(), nn.Linear(200, 1), nn.Dropout())

    def forward(self, x):
        return self.gene1(self.pretrained(x))


def get_res18_dropout(path="../models/res18/tenpercent_resnet18.ckpt"):
    ciga = load_res18_ciga(path)
    return Res18Dropout(ciga)


class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        self.pretrained = models.vgg13()
        self.top_layers = nn.Sequential(nn.Linear(1000, 200), nn.Dropout(), nn.ReLU(), nn.Linear(200, 1), nn.Dropout())

    def forward(self, x):
        return self.top_layers(self.pretrained(x))


def get_vgg13(path=None):
    vgg13 = VGG13()
    if path:
        print(vgg13.load_state_dict(torch.load(path, map_location=torch.device('cpu'))))
    return vgg13
