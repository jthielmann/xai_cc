from torchvision.models import resnet18
from model import load_model
from cluster_functions import get_composite_layertype_layername
import torch.nn as nn


model_dir = "../models/resnet18/RUBCNL/"
model_path = "../models/resnet18/RUBCNL/ep_29.pt"
model = load_model(model_dir, model_path).to("cpu")
model.eval()
model2 = resnet18()
model2.eval()
print(model)
print(model2)
composite, layer_type, layer_name = get_composite_layertype_layername(model)
print(layer_name)
from crp.attribution import CondAttribution
class testmodel(nn.Module):
    def __init__(self):
        super(testmodel, self).__init__()
        self.pretrained = resnet18()
        self.layers = nn.Linear(1000, 10)
    def forward(self, x):
        return self.layers(self.pretrained(x))

model3 = testmodel()

attribution = CondAttribution(model)

from crp.concepts import ChannelConcept
from utils.crp import FeatVis
from script.data_loader import get_dataset_for_plotting

layer_map = {layer_name: ChannelConcept()}
data_dir = "../Training_Data/"
dataset = get_dataset_for_plotting("../Training_Data/")
fv = FeatVis(attribution, dataset, layer_map, preprocess_fn=None, path=f"crp_files/vgg16_imagenet_pexels")
fv.run(composite, 0, len(dataset) // 1, batch_size=32) # needs to be performed once
print("CRP preprocessing done.")