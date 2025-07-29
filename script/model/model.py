import torch.nn as nn
import torch.nn.functional
import torch
import torchvision
from torchvision import models
import json
import timm
import torch.nn.functional as F
import copy
import os
import pandas as pd
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import vgg13,    VGG13_Weights


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.my_new_layers = nn.Sequential(nn.Linear(1000, 200),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(200),
                                           nn.Dropout(0.3),
                                           nn.Linear(200, 10),  #choose x in nn.Linear(20,x) depending on n_classes
                                           nn.ReLU(),
                                           nn.Dropout(0.3))

        self.gene1 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

    def forward(self, x):
        x = self.encoder(x)
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
        self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.gene1 = nn.Sequential(nn.Linear(1000, 200), nn.ReLU(), nn.Linear(200, 1))

    def forward(self, x):
        return self.gene1(self.encoder(x))


def get_res50_mynet2(path=None):
    res50 = MyNet2()
    if path:
        print(res50.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False)))
    return res50


class Res50Dropout(nn.Module):
    def __init__(self, encoder=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)):
        super(Res50Dropout, self).__init__()
        self.encoder = encoder
        self.gene1 = nn.Sequential(nn.Linear(1000, 200), nn.Dropout(),
                                   nn.ReLU(),
                                   nn.Linear(200, 1), nn.Dropout())

    def forward(self, x):
        return self.gene1(self.encoder(x))


def get_res50_dropout(path=None):
    res50 = Res50Dropout()
    if path:
        print(res50.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False)))
    return res50


# old model from jonas
# path = "../../models/res50/10082023_RUBCNL_ST_absolute_single_64_NLR_loss_resnet50.pt"
class MyNet3(nn.Module):
    def __init__(self, my_encoder_model):
        super(MyNet3, self).__init__()
        self.encoder = my_encoder_model

        self.my_new_layers = nn.Sequential(nn.Linear(1000, 200),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(200),
                                           nn.Dropout(0.3),
                                           nn.Linear(200, 1))

    def forward(self, x):
        x = self.encoder(x)
        output = self.my_new_layers(x)
        return output


class Res18_1000(nn.Module):
    def __init__(self, encoder=resnet18(weights=None)):
        super(Res18_1000, self).__init__()
        self.encoder = encoder
        self.gene1 = nn.Sequential(nn.Linear(1000, 200), nn.ReLU(), nn.Linear(200, 1))

    def forward(self, x):
        return self.gene1(self.encoder(x))


def get_res18_1000(path=None):
    res18 = Res18_1000()
    if path:
        print(res18.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False)))
    return res18


class Res18(nn.Module):
    def __init__(self, encoder=resnet18(weights=None)):
        super(Res18, self).__init__()
        self.encoder = encoder
        self.gene1 = nn.Sequential(nn.Linear(512, 200), nn.ReLU(), nn.Linear(200, 1))

    def forward(self, x):
        return self.gene1(self.encoder(x))


def load_res18_ciga(path):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = torchvision.models.__dict__['resnet18'](weights=None)
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
    not_encoder = models.resnet18()
    return Res18_1000(encoder=not_encoder)


class Res50(nn.Module):
    def __init__(self, encoder=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)):
        super(Res50, self).__init__()
        self.encoder = encoder
        self.gene1 = nn.Sequential(nn.Linear(512, 200), nn.ReLU(), nn.Linear(200, 1))

    def forward(self, x):
        return self.gene1(self.encoder(x))


class Res50_1000(nn.Module):
    def __init__(self, encoder=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)):
        super(Res50_1000, self).__init__()
        self.encoder = encoder
        self.gene1 = nn.Sequential(nn.Linear(1000, 200), nn.ReLU(), nn.Linear(200, 1))

    def forward(self, x):
        return self.gene1(self.encoder(x))


def get_res50_random_1000():
    not_encoder = models.resnet50()
    return Res50_1000(encoder=not_encoder)


def get_res50_random():
    not_encoder = models.resnet50()
    return Res50_1000(encoder=not_encoder)


class Res18Dropout(nn.Module):
    def __init__(self, ciga=resnet18(weights=ResNet18_Weights.DEFAULT)):
        super(Res18Dropout, self).__init__()
        self.encoder = ciga
        self.gene1 = nn.Sequential(nn.Linear(512, 200), nn.Dropout(), nn.ReLU(), nn.Linear(200, 1), nn.Dropout())

    def forward(self, x):
        return self.gene1(self.encoder(x))


def init_res18_dropout(path="../models/res18/tenpercent_resnet18.ckpt"):
    ciga = load_res18_ciga(path)
    return Res18Dropout(ciga)


class Res18Dropout_1000(nn.Module):
    def __init__(self, ciga=resnet18(weights=ResNet18_Weights.DEFAULT)):
        super(Res18Dropout_1000, self).__init__()
        self.encoder = ciga
        self.gene1 = nn.Sequential(nn.Linear(1000, 200), nn.Dropout(), nn.ReLU(), nn.Linear(200, 1), nn.Dropout())

    def forward(self, x):
        return self.gene1(self.encoder(x))


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
    def __init__(self, encoder=models.vgg13(weights=VGG13_Weights.IMAGENET1K_V1)):
        super(VGG13, self).__init__()
        self.encoder = encoder
        self.gene1 = nn.Sequential(nn.Linear(1000, 200), nn.ReLU(), nn.Linear(200, 1))

    def forward(self, x):
        return self.gene1(self.encoder(x))


def get_vgg13(path=None):
    vgg13 = VGG13()
    if path:
        print(vgg13.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False)))
    return vgg13


def get_vgg13_random():
    not_encoder = models.vgg13()
    return VGG13(encoder=not_encoder)


class VGG13_Dropout(nn.Module):
    def __init__(self):
        super(VGG13_Dropout, self).__init__()
        self.encoder = models.vgg13(weights=VGG13_Weights.IMAGENET1K_V1)
        self.gene1 = nn.Sequential(nn.Linear(1000, 200), nn.Dropout(), nn.ReLU(), nn.Linear(200, 1), nn.Dropout())

    def forward(self, x):
        return self.gene1(self.encoder(x))


def get_vgg13_dropout(path=None):
    vgg13 = VGG13_Dropout()
    if path:
        print(vgg13.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False)))
    return vgg13


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block_down, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        encoder = []
        encoder.append(nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU()))
        encoder.append(nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        encoder.append(self._make_layer(block_down, 64, layers[0], stride = 1))
        encoder.append(self._make_layer(block_down, 128, layers[1], stride = 2))
        encoder.append(self._make_layer(block_down, 256, layers[2], stride = 2))
        encoder.append(self._make_layer(block_down, 512, layers[3], stride = 2))
        #encoder.append(nn.AvgPool2d(7, stride=1))
        self.encoder = nn.Sequential(*encoder)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, upsample=False):
        super(UpsampleBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, output_padding=(stride - 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.ConvTranspose2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.upsample = None
        if upsample or stride != 1 or in_planes != out_planes:
            # When upsampling, adjust the input dimensions to match the output dimensions.
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, output_padding=(stride - 1), bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(identity)

        out += identity
        out = F.relu(out)

        return out


class ResNetUpsample(nn.Module):
    def __init__(self, block, layers, num_classes=3):
        super(ResNetUpsample, self).__init__()

        # Start with n, 512, 1, 1 input
        self.in_planes = 512

        # Build four layers similar to ResNet
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2, upsample=True)  # Upsample to 7x7
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, upsample=True)  # Upsample to 14x14
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, upsample=True)   # Upsample to 28x28
        self.layer4 = self._make_layer(block, 32, layers[3], stride=2, upsample=True)   # Upsample to 56x56

        # Final layer to upsample to 224x224 and reduce to 3 channels
        self.upsample_final = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=4, padding=0, output_padding=0, bias=False)

    def _make_layer(self, block, planes, blocks, stride=1, upsample=False):
        layers = []
        layers.append(block(self.in_planes, planes, stride, upsample=upsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input: n, 512, 1, 1
        x = self.layer1(x)  # Upsample to n, 256, 7, 7
        x = self.layer2(x)  # Upsample to n, 128, 14, 14
        x = self.layer3(x)  # Upsample to n, 64, 28, 28
        x = self.layer4(x)  # Upsample to n, 32, 56, 56

        # Final upsample to n, 3, 224, 224
        x = self.upsample_final(x)

        return x


def resnet18_upsample(num_classes=3):
    return ResNetUpsample(UpsampleBlock, [2, 2, 2, 2], num_classes)


class ResidualBlock_up(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock_up, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Initial input is (n, 512, 1, 1)
        # self.initial_conv = nn.Upsample(size=(7, 7))  # nn.ConvTranspose2d(512, 512, kernel_size=4, stride=1, padding=0)  # (n, 512, 4, 4)

        # Residual Block 1 (512, 4, 4)
        self.res_block1 = ResidualBlock_up(512)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # (n, 256, 8, 8)

        # Residual Block 2 (256, 8, 8)
        self.res_block2 = ResidualBlock_up(256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # (n, 128, 16, 16)

        # Residual Block 3 (128, 16, 16)
        self.res_block3 = ResidualBlock_up(128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # (n, 64, 32, 32)

        # Residual Block 4 (64, 32, 32)
        self.res_block4 = ResidualBlock_up(64)
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)  # (n, 32, 64, 64)

        # Residual Block 5 (32, 64, 64)
        self.res_block5 = ResidualBlock_up(64)
        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)  # (n, 16, 128, 128)

        # Final upsampling and output (n, 16, 128, 128) -> (n, 3, 224, 224)
        self.final_conv = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)  # (n, 3, 224, 224)

    def forward(self, x):
        # Initial expansion
        #x = F.relu(self.initial_conv(x))

        # Series of residual blocks with upsampling
        x = self.res_block1(x)
        x = F.relu(self.up1(x))

        x = self.res_block2(x)
        x = F.relu(self.up2(x))

        x = self.res_block3(x)
        x = F.relu(self.up3(x))

        x = self.res_block4(x)
        x = F.relu(self.up4(x))

        x = self.res_block5(x)
        x = F.relu(self.up5(x))

        # Final upsampling and output
        x = torch.sigmoid(self.final_conv(x))  # Apply sigmoid to normalize the output to [0, 1]
        return x


class Resnet_ae(nn.Module):
    def __init__(self):
        super(Resnet_ae, self).__init__()
        self.encoder = ResNet(ResidualBlock, [2, 2, 2, 2])
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_Resnet_ae():
    return Resnet_ae()


def get_encoder(path):
    class encoder(nn.Module):
        def __init__(self, path):
            super().__init__()
            ae = get_Resnet_ae()
            if path:
                ae.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False))
            self.encoder = copy.deepcopy(ae.encoder)

        def forward(self, x):
            x = self.encoder(x)
            pool = nn.AvgPool2d(7, stride=1)
            x = pool(x)
            x = x.squeeze()
            return x

    return encoder(path)


class general_model(nn.Module):
    def __init__(self, model_type, gene_list, random_weights=False, dropout=False, dropout_value=None,
                 encoder_path=None, encoder_out_dim=1000, middel_layer_features=200):
        super(general_model, self).__init__()
        if encoder_path and random_weights:
            print("cannot have encoder_path and random_weights set in general_model")
            exit(1)
        if model_type == "vgg13":
            if random_weights:
                weights = None
            else:
                weights = VGG13_Weights.IMAGENET1K_V1
            self.encoder = models.vgg13(weights=weights)
        elif model_type == "resnet18":
            if random_weights:
                weights = None
            else:
                weights = ResNet18_Weights.IMAGENET1K_V1
            self.encoder = models.resnet18(weights=weights)
        elif model_type == "resnet50":
            if random_weights:
                weights = None
            else:
                weights = ResNet50_Weights.IMAGENET1K_V2
            self.encoder = models.resnet50(weights=weights)
        elif model_type == "resnet50d":
            self.encoder = timm.create_model(model_type, num_classes=encoder_out_dim)
        elif model_type == "resnet18d":
            self.encoder = timm.create_model(model_type, num_classes=encoder_out_dim)
        elif model_type == "encoder_res18":
            self.encoder = get_encoder(encoder_path)
        elif model_type == "resnet50dino":
            self.encoder = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        else:
            print("model type", model_type, "not implemented")
            exit(1)
        for gene in gene_list:
            setattr(self, gene, nn.Sequential(nn.Linear(encoder_out_dim, middel_layer_features),nn.ReLU(), nn.Linear(middel_layer_features, 1)))

        self.gene_list = gene_list
        self.model_type = model_type
        self.random_weights = random_weights
        self.dropout = None
        self.encoder_out_dim = encoder_out_dim
        self.dropout_value = dropout_value

    def forward(self, x):
        x = self.encoder(x)
        out = []
        for gene in self.gene_list:
            out.append(getattr(self, gene)(x))
        if len(out) == 1:
            return out[0]
        return torch.cat(out, dim=1)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json_dict = {"model_type": self.model_type, "random_weights": self.random_weights, 'gene_list': self.gene_list,
                         "dropout": self.dropout, "encoder_output_dim": self.encoder_out_dim}
            json.dump(json_dict, f)


def load_model(model_dir, model_name, json_name="settings.json", log_json=False, squelch=False):
    with open(model_dir + "/" + json_name) as f:
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
        if "dropout_value" in d and dropout:
            dropout_value = d["dropout_value"]
        else:
            dropout_value = 0.5
        encoder_out_dim = int(d["encoder_out_dim"])
        if "middel_layer_features" in d:
            middel_layer_features = d["middel_layer_features"]
        else:
            middel_layer_features = 200
    success = False
    model = general_model(model_type, gene_list, random_weights, dropout, dropout_value=dropout_value,
                          encoder_out_dim=encoder_out_dim, middel_layer_features=middel_layer_features)
    try:
        res = model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu'), weights_only=False))
        success = True
        if not squelch:
            print(res)
    except RuntimeError as e:
        pass

    if not success:
        try:
            middel_layer_features = 512
            model = general_model(model_type, gene_list, random_weights, dropout, dropout_value=dropout_value, encoder_out_dim=encoder_out_dim, middel_layer_features=middel_layer_features)
            res = model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu'), weights_only=False))
            success = True
            if not squelch:
                print(res)
        except RuntimeError as e:
            pass
    if not success:
        try:
            middel_layer_features = 1000
            model = general_model(model_type, gene_list, random_weights, dropout, dropout_value=dropout_value, encoder_out_dim=encoder_out_dim, middel_layer_features=middel_layer_features)
            res = model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu'), weights_only=False))
            success = True
            if not squelch:
                print(res)
        except RuntimeError as e:
            if not squelch:
                print(e)
    if not success:
        return None
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


def generate_model_list(model_dir, must_contain=None, model_name = None, skip_names=None, model_list_file_name = "new_models.csv"):
    model_dir_path = []
    for model_type_dir in os.listdir(model_dir):

        sub_path = model_dir + model_type_dir # dirs in "../models/
        if model_type_dir == ".DS_Store" or model_type_dir == "new" or os.path.isfile(sub_path):
            #print("skipping redundant:", model_type_dir)
            continue
        for model_leaf_dir in os.listdir(sub_path):

            sub_path = model_dir + model_type_dir + "/" + model_leaf_dir # e.g. ../models/resnet18/
            if model_type_dir == ".DS_Store" or not os.path.isdir(sub_path) or not os.path.exists(sub_path + "/settings.json"):
                #print("skipping redundant 2:", sub_path)
                continue

            with open(sub_path + "/settings.json") as settings_json:
                d = json.load(settings_json)

                # skip old models
                if "genes" not in d:
                    #print("skipping", sub_path, "because settings.json does not contain: \"genes\"")
                    continue

            if skip_names:
                found_string = False
                for skip_string in skip_names:
                    if sub_path.find(skip_string) != -1:
                        print("skipping", sub_path, "because string contains:", skip_string)
                        found_string = True
                        break
                if found_string:
                    continue
            if must_contain and sub_path.find(must_contain) == -1:
                print("skipping", sub_path, "because string does not contain:", must_contain)
                continue
            if model_name and os.path.exists(sub_path + "/" + model_name):
                model_dir_path.append((sub_path + "/"))
                print("adding:", sub_path + "/" + model_name)
            elif os.path.exists(sub_path + "/best_model.pt"):
                model_dir_path.append((sub_path + "/", sub_path + "/best_model.pt"))
                print("adding:", sub_path + "/best_model.pt")
            elif os.path.exists(sub_path + "/ep_29.pt"):
                model_dir_path.append((sub_path + "/", sub_path + "/ep_29.pt"))
                print("adding:", sub_path + "/ep_29.pt")
            else:
                print("skipping", sub_path, "because it does not contain a model with a fitting name")

    frame = pd.DataFrame(model_dir_path, columns=["model_dir", "model_path"])
    frame.to_csv(model_dir + model_list_file_name, index=False)
    return frame
