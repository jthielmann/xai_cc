import numpy as np
import matplotlib.pyplot as plt
import torch
import zennit
from zennit.torchvision import VGGCanonizer, ResNetCanonizer
from zennit.composites import EpsilonPlusFlat
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.visualization import FeatureVisualization
import torchvision.transforms as T
from PIL import Image
import os

from data_loader import get_dataset_for_plotting, TileLoader
from plot_and_print import plot_tile
from torchvision.models import vgg16
import torch.nn as nn
from model import get_remote_models_and_path

model, model_path = get_remote_models_and_path(model_id=0)
print("model used:", model_path)
model.eval()
#device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"
print("device:", device)
model = model.to(device)
data_dir = "../Training_Data/"
dataset = get_dataset_for_plotting(data_dir)

attribution = CondAttribution(model)

if type(model.pretrained).__name__ == "VGG":
    composite = EpsilonPlusFlat(canonizers=[VGGCanonizer()])
    layer_type = model.pretrained.classifier[-1].__class__
    print(model.pretrained.classifier[-1])
    layer_name = get_layer_names(model, [nn.Linear])[-3]
else:
    composite = EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
    layer_type = model.pretrained.layer1[0].__class__
    # select last bottleneck module
    layer_name = get_layer_names(model, [layer_type])[-1]

print(layer_name)
print(layer_type)
layer_map = {layer_name: ChannelConcept()}

out_path = "../crp_out/tmp"
already_calculated = os.path.exists(out_path)
preprocessing = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=preprocessing, path=out_path)
if not already_calculated:
    fv.run(composite, 0, len(dataset) // 1, batch_size=32)
else:
    print("output folder", out_path, "already exists, assuming CRP is already preprocessed")
print("CRP preprocessing done. output path:", out_path)

from skimage.feature import canny
from scipy.ndimage import filters, gaussian_filter
from crp.image import imgify

img_paths = "../Test_Data/p026/Tiles_156/p026_11_60.tiff"

loader = TileLoader()
data = loader.open(img_paths)
plot_tile(img_paths)

attr = attribution(data.unsqueeze(0).requires_grad_(),
                   [{"y": [0]}],
                   composite,
                   record_layer=[layer_name])

print(f"prediction: class {attr.prediction.argmax().item()} with probability {attr.prediction.softmax(1).max().item():.2f}")


fig, axes = plt.subplots(1, 2, dpi=200, figsize=(3, 2), facecolor='white')
axes[0].imshow(data.permute(1, 2, 0))
axes[0].axis("off")
axes[0].set_title("Test Image")
heatmap = imgify(attr.heatmap[0], cmap="bwr", symmetric=True, level=2.0)

# compute edges for overlay
smooth = data.permute(1, 2, 0).mean(-1)

edges = abs(canny(smooth.numpy(), sigma=1.5, use_quantiles=True))  #, low_threshold=0.7))
edges = gaussian_filter(edges * 1.0, 0.3)
edges = edges / edges.max()
edges[edges > 0.1] = 1

axes[1].imshow(heatmap, alpha=1.0)
axes[1].imshow(1 - edges, alpha=edges*0.3, cmap="gray")

axes[1].axis("off")
axes[1].set_title("LRP Heatmap")
plt.tight_layout()
plt.show()

channel_rels = ChannelConcept().attribute(attr.relevances[layer_name], abs_norm=True) # sum attributions over spatial dimensions and normalize to 1

topk = torch.topk(channel_rels[0], k=5)
topk_ind = topk.indices.detach().cpu().numpy()
topk_rel = channel_rels[0, topk_ind]

print("Top 5 most relevant neurons:")
for neuron, rel in zip(topk_ind, 100 * topk_rel.detach().cpu().numpy()):
    print(f"neuron #{neuron} ({rel:.2f}% relevance)")

