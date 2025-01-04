import torch
print(torch.__version__)

from pcx_utils.render import vis_opaque_img_border
from crp.visualization import FeatureVisualization
import os

import torch

from data_loader import get_dataset_for_plotting
from torchvision.utils import make_grid
import zennit.image as zimage
import pandas as pd
from cluster_functions import (get_umaps, calculate_attributions, load_attributions, get_prototypes)
from model import load_model, generate_model_list
from cluster_functions import get_composite_layertype_layername

import numpy as np
import matplotlib.pyplot as plt

from crp.attribution import CondAttribution

from crp.concepts import ChannelConcept

from crp.image import imgify

import torchvision


device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = "../models/"
model_list_file_name = "new_models.csv"

must_contain = None
update_model_list = True
skip_names = ["AE"]
if not os.path.exists(model_dir + model_list_file_name) or update_model_list:
    print("found these models:")
    frame = generate_model_list(model_dir, must_contain=must_contain, skip_names=skip_names, model_list_file_name=model_list_file_name)
else:
    frame = pd.read_csv(model_dir + model_list_file_name)


from_list = False
for idx, row in frame.iterrows():
    model_dir = row["model_dir"]
    model_path = row["model_path"]
    if model_path.find("dino") == -1:
        continue
    model = load_model(model_dir, model_path, squelch=True).to(device)
    composite, layer_type, layer_name = get_composite_layertype_layername(model)
    model_path = model.model_path

    print("--------------------------------------------------")
    print("model used:", model_path)
    model.eval()
    print("device:", device)
    model = model.to(device)

    data_dir = "../data/jonas/Training_Data/"
    data_dir2 = "../data/CRC-N19/"

    try:
        genes = model.gene_list
    except AttributeError:
        continue

    base_model_dir = "../models"
    out_path = "../crp_out/"
    out_path += model_dir[len(base_model_dir):] + "/"
    token_name = out_path + "clustering_token"
    if os.path.exists(token_name):
        print(token_name, "found, continue..")
        continue

    os.makedirs(out_path, exist_ok=True)
    open(token_name, "a").close()
    print("loading dataset")
    try:
        dataset = get_dataset_for_plotting(data_dir, genes=genes, device=device)
    except KeyError:
        dataset = get_dataset_for_plotting(data_dir2, genes=genes, device=device)

    print("dataset loaded")
    #dataset.dataframe = dataset.dataframe.drop(list(range(10, len(dataset))))

    attribution = CondAttribution(model)

    print("target layer:", layer_name)
    print("target layer type:", layer_type)
    layer_map = {layer_name: ChannelConcept()}


    already_calculated = os.path.exists(out_path) and os.path.exists(out_path + "ActMax_sum_normed/") and os.path.isdir(out_path + "ActMax_sum_normed/")
    #preprocessing = T.Normalize([0.7406, 0.5331, 0.7059], [0.1651, 0.2174, 0.1574])  # from dataloader
    #preprocessing = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # from tutorial
    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=None, path=out_path)
    if not already_calculated:
        print("preprocessing CRP to ", out_path)
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
        activations, attributions, outputs, indices = calculate_attributions(dataset, device, composite, layer_name, attribution, out_path, cc)
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
    """from umap import UMAP

    embedding_attr = UMAP(n_neighbors=5, random_state=123, n_jobs=1)
    X_attr = embedding_attr.fit_transform(attr.detach().cpu().numpy())
    x_attr, y_attr = X_attr[:, 0], X_attr[:, 1]

    embedding_act = UMAP(n_neighbors=5, random_state=123, n_jobs=1)
    X_act = embedding_act.fit_transform(act.detach().cpu().numpy())
    x_act, y_act = X_act[:, 0], X_act[:, 1]"""
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
