from crp_plotting.attribution import CondAttribution
from zennit.attribution import Gradient
from zennit.canonizers import CompositeCanonizer
from crp_plotting.concepts import ChannelConcept
from crp_plotting.helper import get_layer_names
from crp_plotting.visualization import FeatureVisualization
from crp_plotting.image import plot_grid

from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm
import zennit as zen
import torch.nn as nn
import torch
import zennit.torchvision as ztv
from crp_plotting.image import imgify


from relevance import plot_relevance
from model import get_vggs_and_path, get_resnets_and_path, get_remote_models_and_path
from plot_and_print import plot_tile
from data_loader import TileLoader
import os
import pandas as pd
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from data_loader import get_data_loaders, get_dataset, STDataset


import torchvision.transforms as T
from torchvision import transforms

class CRP_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, device="mps", transforms=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # mean and std of the whole dataset
            transforms.Normalize([0.7406, 0.5331, 0.7059], [0.1651, 0.2174, 0.1574])
            ])):
        self.dataframe = dataframe
        self.transforms = transforms
        self.device = device

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        gene_names = list(self.dataframe)[1:]
        gene_vals = []
        row = self.dataframe.iloc[index]
        a = Image.open(row["tile"]).convert("RGB")
        # print(x.size)
        for j in gene_names:
            gene_val = float(row[j])
            gene_vals.append(gene_val)
        e = row["tile"]
        # apply normalization transforms as for encoder colon classifier
        a = self.transforms(a)
        a = a.to(self.device)
        return a, 0


data_dir = "../../Training_Data/"
dataset = CRP_Dataset(get_dataset(data_dir))


models = get_remote_models_and_path()
img_paths = []
img_paths.append("../Test_Data/p026/tiles/p026_11_60.tiff")

loader = TileLoader()
for path in img_paths:
    data = loader.open(path).unsqueeze(0)
    data.requires_grad_(True)
    plot_tile(path)

cc = ChannelConcept()

for model, model_path in models:
    for path in img_paths:
        data = loader.open(path).unsqueeze(0)
        data.requires_grad_(True)
        #print(type(model.encoder))
        if type(model.encoder).__name__ == "VGG":
            composite = zen.composites.EpsilonPlusFlat(canonizers=[ztv.VGGCanonizer()])
        else:
            composite = zen.composites.EpsilonPlusFlat(canonizers=[ztv.ResNetCanonizer()])

        print("selected", type(composite.canonizers[0]).__name__, "for model", model_path)
        attribution = CondAttribution(model, no_param_grad=True)
        #print(model)
        if type(model.encoder).__name__ == "VGG":
            layer_type = model.encoder.classifier[-1].__class__
            print(model.encoder.classifier[-1])
            layer_name = get_layer_names(model, [nn.Linear])[-3]
        else:
            layer_type = model.encoder.layer1[0].__class__
            # select last bottleneck module
            layer_name = get_layer_names(model, [layer_type])[-1]
        #print(layer_type)

        #print(layer_name)
        conditions = [{'y': [0]}]
        attr = attribution(data, conditions, composite, record_layer=[layer_name])

        #print(attr.activations[layer_name].shape, attr.relevances[layer_name].shape)
        # attr[1]["features.40"].shape, attr[2]["features.40"].shape # is equivalent
        rel_c = cc.attribute(attr.relevances[layer_name], abs_norm=True)
        #print(rel_c.shape)
        #print(rel_c)
        rel_values, concept_ids = torch.topk(rel_c[0], 7)
        print(concept_ids, rel_values*100)
        position = model_path.find('/models/') + len('/models/')
        base_path = "../crp_out/" + model_path[position:-3]

        os.makedirs(base_path, exist_ok=True)

        fv_path = base_path
        cc = ChannelConcept()

        layer_names = get_layer_names(model, [layer_type])
        layer_map = {layer : cc for layer in layer_names}
        preprocessing =  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=preprocessing, path=fv_path)
        saved_files = fv.run(composite, 0, len(dataset), 32, 100)
        ref_c = fv.get_max_reference(concept_ids, "encoder.layer4.1", "relevance", (0, 8), composite=composite, plot_fn=None)

        plot_grid(ref_c, figsize=(6, 9))



