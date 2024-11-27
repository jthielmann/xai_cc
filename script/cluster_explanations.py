import numpy as np
import matplotlib.pyplot as plt
import torch
import zennit
from IPython.core.display_functions import display
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.visualization import FeatureVisualization
import torchvision.transforms as T
from PIL import Image
import os

from data_loader import get_dataset_for_plotting, TileLoader, get_patient_loader
from plot_and_print import plot_tile
from torchvision.models import vgg16
import torch.nn as nn
from model import get_remote_models_and_path, load_model
from tqdm import tqdm
from torchvision.utils import make_grid
import zennit.image as zimage
from crp.image import imgify
import pandas as pd
import json

from cluster_functions import (vis_opaque_img_border, get_umaps, calculate_attributions, load_attributions,
                               get_composite_layertype_layername, get_prototypes)


import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)



device = "cpu"
model_dir = "../models/"
# gather new models only
model_dir_path = []
model_list_file_name = "new_models.csv"
update_model_list = True
if not os.path.exists(model_dir + model_list_file_name) or update_model_list:
    for model_type_dir in os.listdir(model_dir):
        sub_path = model_dir + model_type_dir
        if model_type_dir == ".DS_Store" or model_type_dir == "new" or os.path.isfile(sub_path):
            continue
        for model_leaf_dir in os.listdir(sub_path):
            sub_path = model_dir + model_type_dir + "/" + model_leaf_dir
            if model_type_dir == ".DS_Store" or os.path.isfile(sub_path):
                continue
            if not os.path.exists(sub_path + "/settings.json"):
                continue
            with open(sub_path + "/settings.json") as settings_json:
                d = json.load(settings_json)
                model_type = d["model_type"]

                if "genes" not in d:
                    continue

            files = os.listdir(sub_path)
            for f in files:
                if f.find("best_model.pt") != -1:
                    model_dir_path.append((sub_path + "/", sub_path + "/" + os.path.basename(f)))

    frame = pd.DataFrame(model_dir_path, columns=["model_dir", "model_path"])
    frame.to_csv(model_dir + model_list_file_name, index=False)
else:
    frame = pd.read_csv(model_dir + model_list_file_name)


frame = pd.read_csv(model_dir + model_list_file_name)
model = None
i = 0
while True:
    row = frame.iloc[i]
    model = load_model(row["model_dir"], row["model_path"], squelch=True).to(device)
    composite, layer_type, layer_name = get_composite_layertype_layername(model)
    if composite is None:
        continue
    break
model_path = model.model_path
print("model used:", model_path)
model.eval()
#device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)
model = model.to(device)
data_dir = "../Training_Data/"
dataset = get_dataset_for_plotting(data_dir)

attribution = CondAttribution(model)

print("target layer:", layer_name)
print("target layer type:", layer_type)
layer_map = {layer_name: ChannelConcept()}

out_path = "../crp_out/"
already_calculated = os.path.exists(out_path)
preprocessing = T.Normalize([0.7406, 0.5331, 0.7059], [0.1651, 0.2174, 0.1574])  # from dataloader
#preprocessing = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # from tutorial
fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=None, path=out_path)
if not already_calculated:
    fv.run(composite, 0, len(dataset) // 1, batch_size=32)
else:
    print("output folder", out_path, "already exists, assuming CRP is already preprocessed")
print("CRP preprocessing done. output path:", out_path)

cc = ChannelConcept()

activations = []
attributions = []
outputs = []

if not already_calculated or not os.path.exists(out_path + "/activations.pt"):
    print("calculating concept attributions and activations over the dataset")
    activations, attributions, outputs, indices = calculate_attributions(dataset, device, composite, layer_name, out_path, cc)
    print("calculation finished")
else:
    print("loading concept attributions and activations over the dataset")
    activations, attributions, outputs, indices = load_attributions(out_path)


# all indices are 0 because we only have one output
attr = attributions[outputs.argmax(1) == 0]
act = activations[outputs.argmax(1) == 0]
indices = indices[outputs.argmax(1) == 0]

embedding_attr, embedding_act, X_attr, X_act = get_umaps(attr, act, row["model_dir"])
x_attr, y_attr = X_attr[:, 0], X_attr[:, 1]
x_act, y_act = X_act[:, 0], X_act[:, 1]
"""
from scipy import stats
fig, axes = plt.subplots(1, 2, dpi=300, figsize=(5, 3), facecolor='white')
for i, X in enumerate([X_attr, X_act]):
    x, y = X[:, 0], X[:, 1]
    xmin = x.min() - 2
    xmax = x.max() + 2
    ymin = y.min() - 2
    ymax = y.max() + 2
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values, 0.5)
    Z = np.reshape(kernel(positions).T, X.shape).T
    axes[i].contour(Z, extent=[xmin, xmax, ymin, ymax], cmap="Greys", alpha=0.3, extend='min', vmax=Z.max() * 1, zorder=0)
    axes[i].scatter(x, y, s=3, alpha=0.7, zorder=1)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_title(["attributions", "activations"][i])
    axes[i].plot(x[0], y[0], 'ko', markersize=5, label="test image")
    axes[i].legend()
plt.tight_layout()
"""
if not os.path.exists(out_path + "/prototypes/"):
    print("calculating prototypes")
    os.mkdir(out_path + "/prototypes/")
    prototypes = get_prototypes(attr, embedding_attr, act, embedding_act)

    for i in range(len(prototypes)):
        np.save(out_path + "/prototypes/" + str(i), prototypes[i])
else:
    print("loading prototypes")
    prototypes = []
    files = os.listdir(out_path + "/prototypes/")
    for f in files:
        if f.endswith(".npy"):
            prototypes.append(np.load(out_path + "/prototypes/" + f))
proto_attr = prototypes[0]
#exit(0)
print("calculating distances")
distances = np.linalg.norm(attr[:, None, :].detach().cpu() - proto_attr, axis=2)
prototype_samples = np.argsort(distances, axis=0)[:8]
prototype_samples = indices[prototype_samples]

fig, axs = plt.subplots(1, 8, figsize=(6, 8), dpi=200, facecolor='white')


N_PROTOTYPES = 8
for i in range(N_PROTOTYPES):
    imgs_proto = []
    for j in range(8):
        sett = dataset[prototype_samples[j][i]]
        img = sett[0]
        imgs_proto.append(img)

    grid = make_grid(imgs_proto,
        nrow=1,
        padding=0)
    grid = np.array(zimage.imgify(grid.detach().cpu()))
    img = imgify(grid)
    axs[i].imshow(img)
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_title(f"{i}")
plt.show()

import numpy as np
import torch
from crp.image import get_crop_range, imgify




import torchvision

proto = torch.from_numpy(proto_attr)
top_concepts = torch.topk(proto, 3).indices.flatten().unique()
top_concepts = top_concepts[proto[:, top_concepts].amax(0).argsort(descending=True)].tolist()
concept_matrix = proto[:, top_concepts].T
N_CONCEPTS = len(top_concepts)
print("top concepts:", top_concepts)
n_refimgs = 12
ref_imgs = fv.get_max_reference(top_concepts, layer_name, "relevance", (0, 6), composite=composite, rf=True,
                                    plot_fn=vis_opaque_img_border, batch_size=6)

fig, axs = plt.subplots(nrows=N_CONCEPTS + 1, ncols=N_PROTOTYPES + 1, figsize=(N_PROTOTYPES + 6, N_CONCEPTS + 6), dpi=150,
                        gridspec_kw={'width_ratios': [6] + [1 for _ in range(N_PROTOTYPES)],
                                     'height_ratios': [6] + [1 for _ in range(N_CONCEPTS)]})
for i in range(N_CONCEPTS):
    for j in range(N_PROTOTYPES):
        val = concept_matrix[i, j].item()
        axs[i + 1, j + 1].matshow(np.ones((1, 1)) * val if val >= 0 else np.ones((1, 1)) * val * -1,
                                  vmin=0,
                                  vmax=concept_matrix.abs().max(),
                                  cmap="Reds" if val >= 0 else "Blues")
        minmax = concept_matrix.abs().max() * 100 / 2
        cos = val * 100
        color = "white" if abs(cos) > minmax else "black"
        axs[i + 1, j + 1].text(0, 0, f"{cos:.1f}", ha="center", va="center", color=color, fontsize=15)
        axs[i + 1, j + 1].axis('off')
resize = torchvision.transforms.Resize((120, 120))
for i in range(N_PROTOTYPES):
    grid = make_grid(
        [resize(dataset[prototype_samples[j][i]][0])
         for j in range(6)],
        nrow=1,
        padding=0)
    grid = np.array(zimage.imgify(grid.detach().cpu()))
    img = imgify(grid)
    axs[0, i + 1].imshow(img)
    axs[0, i + 1].set_xticks([])
    axs[0, i + 1].set_yticks([])
    axs[0, i + 1].set_title(f"prototype {i}")
    axs[0, 0].axis('off')


for i in range(N_CONCEPTS):
    grid = make_grid(
        [resize(torch.from_numpy(np.asarray(i)).permute((2, 0, 1))) for i in ref_imgs[top_concepts[i]]],
        # [resize(torch.from_numpy(np.asarray(i)).permute((0, 1, 2))) for i in ref_imgs[topk_ind[i]]],
        nrow=int(6 / 1),
        padding=0)
    grid = np.array(zimage.imgify(grid.detach().cpu()))
    axs[i + 1, 0].imshow(grid)
    axs[i + 1, 0].set_ylabel(f"concept {top_concepts[i]}")
    axs[i + 1, 0].set_yticks([])
    axs[i + 1, 0].set_xticks([])

plt.tight_layout()

plt.savefig(out_path + "/result.png")
plt.show()



# top concepts: [1755, 97, 1757, 429, 888, 686, 750]
