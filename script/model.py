import torch.nn as nn
import torch.nn.functional
import torch
import torchvision
from torchvision import models
import json
import timm


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


def get_res50_mynet2(path=None):
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


class auto_encoder(nn.Module):
    def __init__(self, layers, middle_layer_dim, conv=True):
        super(auto_encoder, self).__init__()
        self.encoder = []
        self.decoder = []
        outer_layer_dim = 2^layers * middle_layer_dim
        self.encoder.append(nn.Conv2d(1, outer_layer_dim, 3, stride=2, padding=1))
        self.encoder.append(nn.ReLU(True))
        for i in range(layers)[1:]:
            self.encoder.append(nn.Conv2d(outer_layer_dim, outer_layer_dim, 3, stride=2, padding=1))
            self.encoder.append(nn.ReLU(True))



        self.middle_layer_dim = middle_layer_dim

    def forward(self, x):
        if self.ae:
            x = self.ae(x)
        x = self.pretrained(x)
        out = []
        for gene in self.gene_list:
            out.append(getattr(self, gene)(x))
        return torch.cat(out, dim=1)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json_dict = {"model_type": self.model_type, "random_weights": self.random_weights, 'gene_list': self.gene_list,
                         "dropout": self.dropout, "pretrained_output_dim": self.pretrained_out_dim}
            json.dump(json_dict, f)


"""class ae_res18(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.encoder = models.resnet18()
        self.decoder = []
        for name, param in self.encoder.named_parameters():
            print(name, param.shape)
            self.decoder.append(param)
        print(len(self.decoder))

ae_res18()
exit(0)"""


def get_ae():
    return None


class general_model(nn.Module):
    def __init__(self, model_type, gene_list, random_weights=False, dropout=False, dropout_value=0.5, pretrained_out_dim=1000):
        super(general_model, self).__init__()
        if random_weights:
            weights = None
        else:
            weights='IMAGENET1K_V1'
        if model_type == "vgg13":
            self.pretrained = models.vgg13(weights=weights)
        elif model_type == "resnet18":
            self.pretrained = models.resnet18(weights=weights)
        elif model_type == "resnet50":
            self.pretrained = models.resnet50(weights=weights)
        elif model_type == "resnet50d":
            self.pretrained = timm.create_model(model_type, num_classes=pretrained_out_dim)
        elif model_type == "resnet18d":
            self.pretrained = timm.create_model(model_type, num_classes=pretrained_out_dim)
        else:
            print("model type", model_type, "not implemented")
            exit(1)
        for gene in gene_list:
            if dropout:
                setattr(self, gene, nn.Sequential(nn.Linear(pretrained_out_dim, 200), nn.Dropout(dropout_value), nn.ReLU(), nn.Linear(200, 1), nn.Dropout(dropout_value)))
            else:
                setattr(self, gene, nn.Sequential(nn.Linear(pretrained_out_dim, 200),nn.ReLU(), nn.Linear(200, 1)))

        self.gene_list = gene_list
        self.model_type = model_type
        self.random_weights = random_weights
        self.dropout = dropout
        self.pretrained_out_dim = pretrained_out_dim
        self.ae = get_ae()
        self.dropout_value = dropout_value

    def forward(self, x):
        if self.ae:
            x = self.ae(x)
        x = self.pretrained(x)
        out = []
        for gene in self.gene_list:
            out.append(getattr(self, gene)(x))
        return torch.cat(out, dim=1)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json_dict = {"model_type": self.model_type, "random_weights": self.random_weights, 'gene_list': self.gene_list,
                         "dropout": self.dropout, "pretrained_output_dim": self.pretrained_out_dim}
            json.dump(json_dict, f)


def load_model(model_dir, model_name, json_name="settings.json", log_json=False, squelch=False):
    with open(model_dir + json_name) as f:
        d = json.load(f)
        if log_json:
            print(d)
        model_type = d["model_type"]
        if "genes" in d:
            gene_list = d["genes"]
        else:
            gene_list = [d["gene"]]
        random_weights = d["random_weights"]
        dropout = d["dropout"]
        if "dropout_value" in d:
            dropout_value = d["dropout_value"]
        else:
            dropout_value = 0.5
        pretrained_out_dim = int(d["pretrained_out_dim"])

    model = general_model(model_type, gene_list, random_weights, dropout, dropout_value=dropout_value,pretrained_out_dim=pretrained_out_dim)
    if squelch:
        model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu'), weights_only=False))
    else:
        print(model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu'), weights_only=False)))
    model.model_path = model_name
    return model


def get_resnet_dirs_and_model_names(gene="RUBCNL"):
    paths = []
    paths.append(("../remote_models/new/models/res18/RUBCNL_Res18/", "Res18_1000_ep_29.pt"))
    paths.append(("../remote_models/new/models/res18/RUBCNL_Res18_ciga/", "Res18_ep_29.pt"))
    paths.append(("../remote_models/new/models/res18/RUBCNL_Res18_ciga_drop/", "Res18Dropout_ep_29.pt"))
    paths.append(("../remote_models/new/models/res18/RUBCNL_Res18_freeze/", "Res18_1000_ep_29.pt"))
    paths.append(("../remote_models/new/models/res18/RUBCNL_Res18_random/", "Res18_1000_ep_29.pt"))

    paths.append(("../models/res18/RUBCNL_Res18_ciga/", "Res18_ep_29.pt"))
    paths.append(("../models/res18/RUBCNL_Res18_ciga_drop/", "Res18Dropout_ep_29.pt"))
    paths.append(("../remote_models/new/models/res50/RUBCNL_Res50/", "MyNet2_ep_29.pt"))
    paths.append(("../remote_models/new/models/res50/RUBCNL_Res50_drop/", "Res50Dropout_ep_29.pt"))
    paths.append(("../remote_models/new/models/res50/RUBCNL_Res50_drop_freeze/", "Res50Dropout_ep_29.pt"))
    paths.append(("../remote_models/new/models/res50/RUBCNL_Res50_freeze/", "MyNet2_ep_29.pt"))
    paths.append(("../remote_models/new/models/res50/RUBCNL_Res50_random/", "Res50_1000_ep_29.pt"))
    return paths


def get_vgg_dirs_and_model_names():
    paths = []
    paths.append(("../remote_models/new/models/vgg13/base_model/", "VGG13_ep_29.pt"))
    paths.append(("../remote_models/new/models/vgg13/dropout/", "VGG13_Dropout_ep_29.pt"))
    paths.append(("../remote_models/new/models/vgg13/random_weights/", "VGG13_ep_29.pt"))

def get_resnets_and_path(device="cpu", log_model_name=False, model_id=None):
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

    raw.append(("../models/res50/RUBCNL_HLR_Res50slim/15072024_ep_39_lr_0.0005resnet.pt", get_res50_mynet2))
    raw.append(("../models/res50/RUBCNL_HLR_Res50slim_ice/15072024_ep_39_lr_0.0005resnet.pt", get_res50_mynet2))
    raw.append(("../models/res50/RUBCNL_HLR_Res50slim_optim_ice/RUBCNL_HLR_Res50slim_optim_ice15072024_ep_26_lr_0.0005resnet.pt", get_res50_mynet2))
    raw.append(("../models/res50/RUBCNL_HLR_Res50_Drop/24072024_ep_29_lr_0.0005resnet.pt", get_res50_dropout))
    raw.append(("../remote_models/res50/RUBCNL_Res50/MyNet2_ep_29.pt", get_res50_mynet2))
    raw.append(("../models/res50/RUBCNL_Res50_random/Res50_1000_ep_29.pt", get_res50_mynet2))
    raw.append(("../models/res50/RUBCNL_Res50_freeze/MyNet2_ep_29.pt", get_res50_mynet2))
    raw.append(("../models/res50/RUBCNL_Res50_drop_freeze/Res50Dropout_ep_29.pt", get_res50_dropout))

    if model_id is None:
        models = []
        for r in raw:
            if log_model_name:
                print(r[0])
            models.append((r[1](r[0]).to(device).eval(), r[0]))

        return models
    else:
        r = raw[model_id]
        return r[1](r[0]).to(device).eval(), r[0]


def get_vggs_and_path(device="cpu", log_model_name=False, model_id=None):
    raw = []
    raw.append(("../models/vgg13/dropout/VGG13_Dropout_ep_29.pt", get_vgg13_dropout))
    raw.append(("../models/vgg13/base_model/VGG13_ep_29.pt", get_vgg13))
    raw.append(("../models/vgg13/random_weights/VGG13_ep_29.pt", get_vgg13))

    if model_id is None:
        models = []
        for r in raw:
            if log_model_name:
                print(r[0])
            models.append((r[1](r[0]).to(device).eval(), r[0]))

        return models
    else:
        r = raw[model_id]
        return r[1](r[0]).to(device).eval(), r[0]


def get_models_and_path(device="cpu", log_model_name=False, model_id=None):
    raw = []
    raw.append(("../models/res18/RUBCNL_Res18/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../models/res18/res18_ciga_freeze/Res18_ep_29.pt", get_res18_ciga))
    raw.append(("../models/res18/RUBCNL_Res18_freeze/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../models/res18/RUBCNL_Res18_random/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../models/res18/RUBCNL_Res18_ciga/Res18_ep_29.pt", get_res18_ciga))
    raw.append(("../models/res18/RUBCNL_Res18_ciga_drop/Res18Dropout_ep_29.pt", get_res18_ciga_dropout))
    #raw.append(("../models/res18/res18_iced_e29.pt",get_res18_ciga))
    #raw.append(("../models/res18/res18_not_iced_e29.pt", get_res18_ciga))

    #raw.append(("../models/res50/RUBCNL_HLR_Res50slim/15072024_ep_39_lr_0.0005resnet.pt", get_res50))
    #raw.append(("../models/res50/RUBCNL_HLR_Res50slim_ice/15072024_ep_39_lr_0.0005resnet.pt", get_res50))
    #raw.append(("../models/res50/RUBCNL_HLR_Res50slim_optim_ice/RUBCNL_HLR_Res50slim_optim_ice15072024_ep_26_lr_0.0005resnet.pt", get_res50))
    #raw.append(("../models/res50/RUBCNL_HLR_Res50_Drop/24072024_ep_29_lr_0.0005resnet.pt", get_res50_dropout))
    #raw.append(("../remote_models/res50/RUBCNL_Res50/MyNet2_ep_29.pt", get_res50))
    raw.append(("../models/res50/RUBCNL_Res50_random/Res50_1000_ep_29.pt", get_res50_mynet2))
    raw.append(("../models/res50/RUBCNL_Res50_freeze/MyNet2_ep_29.pt", get_res50_mynet2))
    raw.append(("../models/res50/RUBCNL_Res50_drop_freeze/Res50Dropout_ep_29.pt", get_res50_dropout))

    raw.append(("../models/vgg13/dropout/VGG13_Dropout_ep_29.pt", get_vgg13_dropout))
    raw.append(("../models/vgg13/base_model/VGG13_ep_29.pt", get_vgg13))
    raw.append(("../models/vgg13/random_weights/VGG13_ep_29.pt", get_vgg13))

    if model_id is None:
        models = []
        for r in raw:
            if log_model_name:
                print(r[0])
            models.append((r[1](r[0]).to(device).eval(), r[0]))

        return models
    else:
        r = raw[model_id]
        return r[1](r[0]).to(device).eval(), r[0]


def get_remote_resnets_and_path(device="cpu", log_model_name=False, model_id=None):
    raw = []
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_ciga/Res18_ep_29.pt", get_res18_ciga))
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_ciga_drop/Res18Dropout_ep_29.pt", get_res18_ciga_dropout))
    #raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_drop/Res18Dropout_ep_29.pt", get_res18_ciga_dropout))
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_freeze/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_random/Res18_1000_ep_29.pt", get_res18_1000))

    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50/MyNet2_ep_9.pt", get_res50_mynet2))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50/MyNet2_ep_19.pt", get_res50_mynet2))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50/MyNet2_ep_29.pt", get_res50_mynet2))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50/MyNet2_ep_39.pt", get_res50_mynet2))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50_drop/Res50Dropout_ep_29.pt", get_res50_dropout))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50_drop_freeze/Res50Dropout_ep_29.pt", get_res50_dropout))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50_freeze/MyNet2_ep_29.pt", get_res50_mynet2))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50_random/Res50_1000_ep_29.pt", get_res50_mynet2))

    if model_id is None:
        models = []
        for r in raw:
            if log_model_name:
                print(r[0])
            models.append((r[1](r[0]).to(device).eval(), r[0]))

        return models
    else:
        r = raw[model_id]
        return r[1](r[0]).to(device).eval(), r[0]


def get_remote_vggs_and_path(device="cpu", log_model_name=False, model_id=None):
    raw = []

    raw.append(("../remote_models/new/models/vgg13/base_model/VGG13_ep_29.pt", get_vgg13))
    raw.append(("../remote_models/new/models/vgg13/dropout/VGG13_Dropout_ep_29.pt", get_vgg13_dropout))
    raw.append(("../remote_models/new/models/vgg13/random_weights/VGG13_ep_29.pt", get_vgg13))
    if model_id is None:
        models = []
        for r in raw:
            if log_model_name:
                print(r[0])
            models.append((r[1](r[0]).to(device).eval(), r[0]))

        return models
    else:
        r = raw[model_id]
        return r[1](r[0]).to(device).eval(), r[0]


def get_remote_models_and_path(device="cpu", log_model_name=False, model_id=None):
    raw = []
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_ciga/Res18_ep_29.pt", get_res18_ciga))
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_ciga_drop/Res18Dropout_ep_29.pt", get_res18_ciga_dropout))
    #raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_drop/Res18Dropout_ep_29.pt", get_res18_ciga_dropout))
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_freeze/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../remote_models/new/models/res18/RUBCNL_Res18_random/Res18_1000_ep_29.pt", get_res18_1000))

    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50/MyNet2_ep_29.pt", get_res50_mynet2))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50_drop/Res50Dropout_ep_29.pt", get_res50_dropout))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50_drop_freeze/Res50Dropout_ep_29.pt", get_res50_dropout))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50_freeze/MyNet2_ep_29.pt", get_res50_mynet2))
    raw.append(("../remote_models/new/models/res50/RUBCNL_Res50_random/Res50_1000_ep_29.pt", get_res50_mynet2))

    raw.append(("../remote_models/new/models/vgg13/base_model/VGG13_ep_29.pt", get_vgg13))
    raw.append(("../remote_models/new/models/vgg13/dropout/VGG13_Dropout_ep_29.pt", get_vgg13_dropout))
    raw.append(("../remote_models/new/models/vgg13/random_weights/VGG13_ep_29.pt", get_vgg13))
    if model_id is None:
        models = []
        for r in raw:
            if log_model_name:
                print(r[0])
            models.append((r[1](r[0]).to(device).eval(), r[0]))

        return models
    else:
        r = raw[model_id]
        return r[1](r[0]).to(device).eval(), r[0]


def get_remote_models_and_path_mki67(device="cpu", log_model_name=False, model_id=None):
    raw = []
    raw.append(("../remote_models/new/models/res18/MKI67_Res18/Res18_1000_ep_29.pt", get_res18_1000))
    raw.append(("../remote_models/new/models/res50/MKI67_Res50/MyNet2_ep_29.pt", get_res50_mynet2))
    if model_id is None:
        models = []
        for r in raw:
            if log_model_name:
                print(r[0])
            models.append((r[1](r[0]).to(device).eval(), r[0]))

        return models
    else:
        r = raw[model_id]
        return r[1](r[0]).to(device).eval(), r[0]
