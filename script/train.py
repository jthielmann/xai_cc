import torch.nn as nn
import torchvision
from torchvision import models
import torch
from func import get_model, get_model2, training


def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "../models/res18/tenpercent_resnet18.ckpt"
model = torchvision.models.__dict__['resnet18'](pretrained=False)
state = torch.load(MODEL_PATH, map_location=device)

state_dict = state['state_dict']
print()
for key in list(state_dict.keys()):
    state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

ciga = load_model_weights(model, state_dict)
ciga.fc = torch.nn.Sequential()

class Res18(nn.Module):
    def __init__(self, ciga):
        super(Res18, self).__init__()
        self.pretrained = ciga

        self.gene1 = nn.Sequential(nn.Linear(512, 200), nn.ReLU(), nn.Linear(200, 1))

    def forward(self, x):
        x = self.pretrained(x)
        x = self.gene1(x)
        return x


# for logging purposes
training(resnet=Res18(ciga),
         data_dir='../Training_Data/',
         model_save_dir="./RUBCNL_HLR_Res18/",
         epochs=30,
         loss_fn=nn.MSELoss(),
         learning_mode="HLR",
         batch_size=64,
         gene="RUBCNL")