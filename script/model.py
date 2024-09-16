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
        print(model.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False)))
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
        print(res50.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False)))
    return res50


class Res50Dropout(nn.Module):
    def __init__(self, pretrained=models.resnet50(weights="IMAGENET1K_V2")):
        super(Res50Dropout, self).__init__()
        self.pretrained = pretrained
        self.gene1 = nn.Sequential(nn.Linear(1000, 200), nn.Dropout(),
                                   nn.ReLU(),
                                   nn.Linear(200, 1), nn.Dropout())

    def forward(self, x):
        return self.gene1(self.pretrained(x))


def get_res50_dropout(path=None):
    res50 = Res50Dropout()
    if path:
        print(res50.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False)))
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


class Res18_1000(nn.Module):
    def __init__(self, pretrained=models.resnet18(weights="DEFAULT")):
        super(Res18_1000, self).__init__()
        self.pretrained = pretrained
        self.gene1 = nn.Sequential(nn.Linear(1000, 200), nn.ReLU(), nn.Linear(200, 1))

    def forward(self, x):
        return self.gene1(self.pretrained(x))


def get_res18_1000(path=None):
    res18 = Res18_1000()
    if path:
        print(res18.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False)))
    return res18


class Res18(nn.Module):
    def __init__(self, pretrained=models.resnet18(weights="DEFAULT")):
        super(Res18, self).__init__()
        self.pretrained = pretrained
        self.gene1 = nn.Sequential(nn.Linear(512, 200), nn.ReLU(), nn.Linear(200, 1))

    def forward(self, x):
        return self.gene1(self.pretrained(x))


def load_res18_ciga(path):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = torchvision.models.__dict__['resnet18'](pretrained=False)
    state = torch.load(path, map_location=device, weights_only=False)

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
    print(model.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False)))
    return model


def init_res18_ciga_dropout(path):
    ciga = load_res18_ciga(path)

    return Res18Dropout(ciga)


def get_res18_ciga_dropout(path):
    ciga = models.resnet18()
    ciga.fc = torch.nn.Sequential()
    model = Res18Dropout(ciga)
    print(model.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False)))
    return model



def get_res18(path=None):
    res18 = Res18()
    if path:
        print(res18.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False)))
    return res18


def get_res18_random():
    not_pretrained = models.resnet18()
    return Res18_1000(pretrained=not_pretrained)


class Res50(nn.Module):
    def __init__(self, pretrained=models.resnet50(weights="IMAGENET1K_V2")):
        super(Res50, self).__init__()
        self.pretrained = pretrained
        self.gene1 = nn.Sequential(nn.Linear(512, 200), nn.ReLU(), nn.Linear(200, 1))

    def forward(self, x):
        return self.gene1(self.pretrained(x))


class Res50_1000(nn.Module):
    def __init__(self, pretrained=models.resnet50(weights="IMAGENET1K_V2")):
        super(Res50_1000, self).__init__()
        self.pretrained = pretrained
        self.gene1 = nn.Sequential(nn.Linear(1000, 200), nn.ReLU(), nn.Linear(200, 1))

    def forward(self, x):
        return self.gene1(self.pretrained(x))


def get_res50_random_1000():
    not_pretrained = models.resnet50()
    return Res50_1000(pretrained=not_pretrained)


def get_res50_random():
    not_pretrained = models.resnet50()
    return Res50_1000(pretrained=not_pretrained)


class Res18Dropout(nn.Module):
    def __init__(self, ciga=models.resnet18(weights='DEFAULT')):
        super(Res18Dropout, self).__init__()
        self.pretrained = ciga
        self.gene1 = nn.Sequential(nn.Linear(512, 200), nn.Dropout(), nn.ReLU(), nn.Linear(200, 1), nn.Dropout())

    def forward(self, x):
        return self.gene1(self.pretrained(x))


def init_res18_dropout(path="../models/res18/tenpercent_resnet18.ckpt"):
    ciga = load_res18_ciga(path)
    return Res18Dropout(ciga)


class Res18Dropout_1000(nn.Module):
    def __init__(self, ciga=models.resnet18(weights='DEFAULT')):
        super(Res18Dropout_1000, self).__init__()
        self.pretrained = ciga
        self.gene1 = nn.Sequential(nn.Linear(1000, 200), nn.Dropout(), nn.ReLU(), nn.Linear(200, 1), nn.Dropout())

    def forward(self, x):
        return self.gene1(self.pretrained(x))


def get_res18_dropout_1000(path=None):
    res18 = Res18Dropout()
    if path:
        print(res18.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False)))
    return res18


def get_res18_dropout(path=None):
    res18 = Res18Dropout()
    if path:
        print(res18.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False)))
    return res18


class VGG13(nn.Module):
    def __init__(self, pretrained=models.vgg13(weights='IMAGENET1K_V1')):
        super(VGG13, self).__init__()
        self.pretrained = pretrained
        self.gene1 = nn.Sequential(nn.Linear(1000, 200), nn.ReLU(), nn.Linear(200, 1))

    def forward(self, x):
        return self.gene1(self.pretrained(x))


def get_vgg13(path=None):
    vgg13 = VGG13()
    if path:
        print(vgg13.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False)))
    return vgg13


def get_vgg13_random():
    not_pretrained = models.vgg13()
    return VGG13(pretrained=not_pretrained)


class VGG13_Dropout(nn.Module):
    def __init__(self):
        super(VGG13_Dropout, self).__init__()
        self.pretrained = models.vgg13(weights='IMAGENET1K_V1')
        self.gene1 = nn.Sequential(nn.Linear(1000, 200), nn.Dropout(), nn.ReLU(), nn.Linear(200, 1), nn.Dropout())

    def forward(self, x):
        return self.gene1(self.pretrained(x))


def get_vgg13_dropout(path=None):
    vgg13 = VGG13_Dropout()
    if path:
        print(vgg13.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False)))
    return vgg13


def get_resnets_and_path(device="cpu", log_model_name=False):
    raw = []
    raw.append(("../models/res18/RUBCNL_Res18/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../models/res18/res18_ciga_freeze/Res18_ep_29.pt", get_res18_ciga))
    raw.append(("../models/res18/RUBCNL_Res18_freeze/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../models/res18/RUBCNL_Res18/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../models/res18/RUBCNL_Res18_random/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../models/res18/RUBCNL_Res18_ciga/Res18_ep_29.pt", get_res18_ciga))
    raw.append(("../models/res18/RUBCNL_Res18_ciga_drop/Res18Dropout_ep_29.pt", get_res18_ciga_dropout))
    raw.append(("../models/res18/res18_iced_e29.pt",get_res18_ciga))
    raw.append(("../models/res18/res18_not_iced_e29.pt", get_res18_ciga))

    raw.append(("../models/res50/RUBCNL_HLR_Res50slim/15072024_ep_39_lr_0.0005resnet.pt", get_res50))
    raw.append(("../models/res50/RUBCNL_HLR_Res50slim_ice/15072024_ep_39_lr_0.0005resnet.pt", get_res50))
    raw.append(("../models/res50/RUBCNL_HLR_Res50slim_optim_ice/RUBCNL_HLR_Res50slim_optim_ice15072024_ep_26_lr_0.0005resnet.pt", get_res50))
    raw.append(("../models/res50/RUBCNL_HLR_Res50_Drop/24072024_ep_29_lr_0.0005resnet.pt", get_res50_dropout))
    raw.append(("../remote_models/res50/RUBCNL_Res50/MyNet2_ep_29.pt", get_res50))
    raw.append(("../models/res50/RUBCNL_Res50_random/Res50_1000_ep_29.pt", get_res50))
    raw.append(("../models/res50/RUBCNL_Res50_freeze/MyNet2_ep_29.pt", get_res50))
    raw.append(("../models/res50/RUBCNL_Res50_drop_freeze/Res50Dropout_ep_29.pt", get_res50_dropout))

    models = []
    for r in raw:
        if log_model_name:
            print(r[0])
        models.append((r[1](r[0]).to(device).eval(), r[0]))
    return models


def get_vggs_and_path(device="cpu", log_model_name=False):
    raw = []
    raw.append(("../models/vgg13/dropout/VGG13_Dropout_ep_29.pt", get_vgg13_dropout))
    raw.append(("../models/vgg13/base_model/VGG13_ep_29.pt", get_vgg13))
    raw.append(("../models/vgg13/random_weights/VGG13_ep_29.pt", get_vgg13))

    models = []
    for r in raw:
        if log_model_name:
            print(r[0])
        models.append((r[1](r[0]).to(device).eval(), r[0]))
    return models


def get_models_and_path(device="cpu", log_model_name=False):
    raw = []
    raw.append(("../models/res18/RUBCNL_Res18/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../models/res18/res18_ciga_freeze/Res18_ep_29.pt", get_res18_ciga))
    raw.append(("../models/res18/RUBCNL_Res18_freeze/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../models/res18/RUBCNL_Res18_random/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../models/res18/RUBCNL_Res18_ciga/Res18_ep_29.pt", get_res18_ciga))
    raw.append(("../models/res18/RUBCNL_Res18_ciga_drop/Res18Dropout_ep_29.pt", get_res18_ciga_dropout))
    raw.append(("../models/res18/res18_iced_e29.pt",get_res18_ciga))
    raw.append(("../models/res18/res18_not_iced_e29.pt", get_res18_ciga))

    raw.append(("../models/res50/RUBCNL_HLR_Res50slim/15072024_ep_39_lr_0.0005resnet.pt", get_res50))
    raw.append(("../models/res50/RUBCNL_HLR_Res50slim_ice/15072024_ep_39_lr_0.0005resnet.pt", get_res50))
    raw.append(("../models/res50/RUBCNL_HLR_Res50slim_optim_ice/RUBCNL_HLR_Res50slim_optim_ice15072024_ep_26_lr_0.0005resnet.pt", get_res50))
    raw.append(("../models/res50/RUBCNL_HLR_Res50_Drop/24072024_ep_29_lr_0.0005resnet.pt", get_res50_dropout))
    raw.append(("../remote_models/res50/RUBCNL_Res50/MyNet2_ep_29.pt", get_res50))
    raw.append(("../models/res50/RUBCNL_Res50_random/Res50_1000_ep_29.pt", get_res50))
    raw.append(("../models/res50/RUBCNL_Res50_freeze/MyNet2_ep_29.pt", get_res50))
    raw.append(("../models/res50/RUBCNL_Res50_drop_freeze/Res50Dropout_ep_29.pt", get_res50_dropout))

    raw.append(("../models/vgg13/dropout/VGG13_Dropout_ep_29.pt", get_vgg13_dropout))
    raw.append(("../models/vgg13/base_model/VGG13_ep_29.pt", get_vgg13))
    raw.append(("../models/vgg13/random_weights/VGG13_ep_29.pt", get_vgg13))
    models = []
    for r in raw:
        if log_model_name:
            print(r[0])
        models.append((r[1](r[0]).to(device).eval(), r[0]))
    return models


def get_remote_resnets_and_path(device="cpu", log_model_name=False):
    raw = []
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_ciga/Res18_ep_29.pt", get_res18_ciga))
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_ciga_drop/Res18Dropout_ep_29.pt", get_res18_ciga_dropout))
    #raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_drop/Res18Dropout_ep_29.pt", get_res18_ciga_dropout))
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_freeze/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_random/Res18_1000_ep_29.pt", get_res18_1000))

    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50/MyNet2_ep_9.pt", get_res50))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50/MyNet2_ep_19.pt", get_res50))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50/MyNet2_ep_29.pt", get_res50))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50/MyNet2_ep_39.pt", get_res50))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50_drop/Res50Dropout_ep_29.pt", get_res50_dropout))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50_drop_freeze/Res50Dropout_ep_29.pt", get_res50_dropout))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50_freeze/MyNet2_ep_29.pt", get_res50))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50_random/Res50_1000_ep_29.pt", get_res50))

    models = []
    for r in raw:
        if log_model_name:
            print(r[0])
        models.append((r[1](r[0]).to(device).eval(), r[0]))
    return models


def get_remote_vggs_and_path(device="cpu", log_model_name=False):
    raw = []

    raw.append(("../remote_models/new/models/vgg13/base_model/VGG13_ep_29.pt", get_vgg13))
    raw.append(("../remote_models/new/models/vgg13/dropout/VGG13_Dropout_ep_29.pt", get_vgg13_dropout))
    raw.append(("../remote_models/new/models/vgg13/random_weights/VGG13_ep_29.pt", get_vgg13))
    models = []
    for r in raw:
        if log_model_name:
            print(r[0])
        models.append((r[1](r[0]).to(device).eval(), r[0]))
    return models


def get_remote_models_and_path(device="cpu", log_model_name=False):
    raw = []
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_ciga/Res18_ep_29.pt", get_res18_ciga))
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_ciga_drop/Res18Dropout_ep_29.pt", get_res18_ciga_dropout))
    #raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_drop/Res18Dropout_ep_29.pt", get_res18_ciga_dropout))
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_freeze/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_random/Res18_1000_ep_29.pt", get_res18_1000))

    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50/MyNet2_ep_29.pt", get_res50))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50_drop/Res50Dropout_ep_29.pt", get_res50_dropout))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50_drop_freeze/Res50Dropout_ep_29.pt", get_res50_dropout))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50_freeze/MyNet2_ep_29.pt", get_res50))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50_random/Res50_1000_ep_29.pt", get_res50))

    raw.append(("../remote_models/new/models/vgg13/base_model/VGG13_ep_29.pt", get_vgg13))
    raw.append(("../remote_models/new/models/vgg13/dropout/VGG13_Dropout_ep_29.pt", get_vgg13_dropout))
    raw.append(("../remote_models/new/models/vgg13/random_weights/VGG13_ep_29.pt", get_vgg13))
    models = []
    for r in raw:
        if log_model_name:
            print(r[0])
        models.append((r[1](r[0]).to(device).eval(), r[0]))
    return models


def get_remote_models_and_path_mki67(device="cpu", log_model_name=False):
    raw = []
    raw.append(("../remote_models/new/models/res18/MKI67_Res18/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../remote_models/new/models/res50/MKI67_Res50/MyNet2_ep_29.pt", get_res50))
    models = []
    for r in raw:
        if log_model_name:
            print(r[0])
        models.append((r[1](r[0]).to(device).eval(), r[0]))
    return models
