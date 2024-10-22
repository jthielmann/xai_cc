import torch.nn as nn
import torch.nn.functional
import torch
import torchvision
from pandas.core.dtypes.common import ensure_object
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


import torch
import torch.nn as nn

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

"""
class ResidualBlock_upsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, upsample = None):
        super(ResidualBlock_upsample, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 0),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 0),
                        nn.BatchNorm2d(out_channels))
        self.upsample = upsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        print("residual shape before:", residual.shape)
        if self.upsample:
            residual = self.upsample(x)
        print("residual shape after:", residual.shape)
        print("out shape after:", out.shape)
        out += residual
        out = self.relu(out)
        return out
"""

class ResNet(nn.Module):
    def __init__(self, block_down, block_up, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        encoder = []
        encoder.append(nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU()))
        encoder.append(nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        encoder.append(self._make_layer_down(block_down, 64, layers[0], stride = 1))
        encoder.append(self._make_layer_down(block_down, 128, layers[1], stride = 2))
        encoder.append(self._make_layer_down(block_down, 256, layers[2], stride = 2))
        encoder.append(self._make_layer_down(block_down, 512, layers[3], stride = 2))
        encoder.append(nn.AvgPool2d(7, stride=1))
        self.encoder = nn.Sequential(*encoder)
        decoder = []
        decoder.append(nn.Upsample(size=(7,7)))
        decoder.append(self._make_layer_up(block_up, 512, layers[0], stride = 2))

        self.decoder = nn.Sequential(*decoder)

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

    """
    def _make_layer_up(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes:

            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes, kernel_size=5, stride=stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    """
    def forward(self, x):
        encoder_shapes = []
        for l in self.encoder:
            x = l(x)
            encoder_shapes.append(x.shape)
        print(encoder_shapes)
        # 1, 512, 1, 1 to 1, 512
        #x = x.view(x.size(0), -1)
        decoder_shapes = []

        decoder_shapes.append(x.shape)
        for l in self.decoder:
            x = l(x)
            decoder_shapes.append(x.shape)
        print(len(decoder_shapes))
        for i in range(len(decoder_shapes)):
            print("decoder shape", i, decoder_shapes[i])
            print("encoder shape", i, encoder_shapes[-(i+1)])
        x = self.decoder(x)

        return x


r = ResNet(ResidualBlock,[2, 2, 2, 2])
input = torch.rand(1, 3, 224, 224)
out = r(input)
exit(0)
def get_ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])


import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Example usage:
model = resnet18_upsample(num_classes=3)
input_tensor = torch.randn(1, 512, 1, 1)  # Batch size of 1 with 512x1x1 input
output = model(input_tensor)
print(output.shape)  # Expected shape: (1, 3, 224, 224)




#-----------------------------------------------------------------------------------------------------------------------







class resnet(nn.Module):
    def __init__(self, pretrained=models.resnet18(weights='DEFAULT')):
        super(resnet, self).__init__()


    def forward(self, x):
        return self.gene1(self.pretrained(x))




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
