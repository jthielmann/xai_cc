print("fahrkarte0")
from umap import UMAP
from cluster_functions import calculate_attributions, load_attributions, get_composite_layertype_layername
from data_loader import get_dataset_for_plotting
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.visualization import FeatureVisualization
import torchvision.transforms as T
from model import load_model
import os

data_dir = "../Training_Data/"
#dataset = get_dataset_for_plotting(data_dir)
from datasets import get_dataset
from torchvision.models import vgg16

dataset_fn = get_dataset("imagenet_pexels")
dataset = dataset_fn(data_path="datasets/pexels", preprocessing=True, split="train")
device = "cpu"

model_dir = "../models/resnet18/RUBCNL/"
model_path = "../models/resnet18/RUBCNL/ep_29.pt"
model = load_model(model_dir, model_path).to(device)
model.eval()
#device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

attribution = CondAttribution(model)
attribution
composite, layer_type, layer_name = get_composite_layertype_layername(model)
print("target layer:", layer_name)
print("target layer type:", layer_type)
layer_map = {layer_name: ChannelConcept()}

out_path = "../crp_out/test"
already_calculated = os.path.exists(out_path) and os.path.exists(out_path + "ActMax_sum_normed/") and os.listdir(out_path + "ActMax_sum_normed/")
preprocessing = T.Normalize([0.7406, 0.5331, 0.7059], [0.1651, 0.2174, 0.1574])  # from dataloader
#preprocessing = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # from tutorial
print("fahrkarte2")
fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=None, path=out_path)
fv.run(composite, 0, len(dataset) // 1, batch_size=32) # needs to be performed once
print("fahrkarte3")

if not already_calculated or not os.path.exists(out_path + "/activations.pt"):
    print("calculating concept attributions and activations over the dataset")
    cc = ChannelConcept()

    activations, attributions, outputs, indices = calculate_attributions(dataset, device, composite, layer_name, attribution, out_path, cc)
    print("calculation finished")
else:
    print("loading concept attributions and activations over the dataset")
    activations, attributions, outputs, indices = load_attributions(out_path)

print("fahrkarte end")
# all indices are 0 because we only have one output
attr = attributions[outputs.argmax(1) == 0]
act = activations[outputs.argmax(1) == 0]
indices = indices[outputs.argmax(1) == 0]

umap = UMAP(n_neighbors=5, random_state=123, n_jobs=1)
umap.fit_transform(attr.detach().cpu().numpy())