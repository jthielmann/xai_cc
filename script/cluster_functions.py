import os.path

import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import DataLoader
from PIL import ImageFilter, Image, ImageDraw
from crp.image import get_crop_range, imgify
from torchvision.transforms.functional import gaussian_blur
from zennit.core import stabilize
from umap import UMAP
from tqdm import tqdm
from crp.helper import get_layer_names

from zennit.torchvision import VGGCanonizer, ResNetCanonizer
from zennit.composites import EpsilonPlusFlat
import torch.nn as nn
from sklearn.mixture import GaussianMixture

from pcx_utils.render import vis_opaque_img_border
from crp.visualization import FeatureVisualization
import os
from data_loader import get_dataset_for_plotting
from torchvision.utils import make_grid
import zennit.image as zimage
import pandas as pd
from model import load_model, generate_model_list

import matplotlib.pyplot as plt

from crp.attribution import CondAttribution

from crp.concepts import ChannelConcept

from crp.image import imgify

import torchvision


def get_composite_layertype_layername(model):
    if type(model.pretrained).__name__ == "VGG":
        composite = EpsilonPlusFlat(canonizers=[VGGCanonizer()])
        layer_type = model.pretrained.classifier[-1].__class__
        print(model.pretrained.classifier[-1])
        layer_name = get_layer_names(model, [nn.Linear])[-3]
    elif type(model.pretrained).__name__ == "encoder":
        return None, None, None
        composite = EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
        layer_type = model.pretrained.encoder.encoder[-1][-1].conv2[0].__class__
        layer_name = get_layer_names(model, [layer_type])[-1]
    else:
        composite = EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
        layer_type = model.pretrained.layer1[0].__class__
        # select last bottleneck module
        layer_name = get_layer_names(model, [layer_type])[-1]
    return composite, layer_type, layer_name


def get_composite_layertype_layername_lightning(model):
    composite = EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
    layer_type = model.encoder.layer1[0].__class__
    # select last bottleneck module
    layer_name = get_layer_names(model, [layer_type])[-1]
    return composite, layer_type, layer_name


def calculate_attributions(dataloader, device, composite, layer_name, attribution, out_path, cc):
    activations = []
    attributions = []
    outputs = []
    for i, (x, _) in enumerate(tqdm(dataloader)):
        x = x.to(device).unsqueeze(0).requires_grad_()
        condition = [{"y": [0]}]
        attr = attribution(x, condition, composite, record_layer=[layer_name])

        attributions.append(cc.attribute(attr.relevances[layer_name], abs_norm=True))
        activations.append(attr.activations[layer_name].amax((-2, -1)))
        outputs.extend([attr.prediction.detach().cpu()])

    activations = torch.cat(activations)
    torch.save(activations, out_path + "/activations.pt")
    attributions = torch.cat(attributions)
    torch.save(attributions, out_path + "/attributions.pt")
    outputs = torch.cat(outputs)
    torch.save(outputs, out_path + "/outputs.pt")
    indices = np.arange(len(dataloader.dataset))
    torch.save(indices, out_path + "/indices.pt")
    return activations, attributions, outputs, indices


def load_attributions(out_path):
    activations = torch.load(out_path + "/activations.pt", weights_only=False, map_location='cpu')
    attributions = torch.load(out_path + "/attributions.pt", weights_only=False, map_location='cpu')
    outputs = torch.load(out_path + "/outputs.pt", weights_only=False, map_location='cpu')
    indices = torch.load(out_path + "/indices.pt", weights_only=False, map_location='cpu')
    return activations, attributions, outputs, indices


def mystroke(img, size: int, color: str = 'black'):

    X, Y = img.size
    edge = img.filter(ImageFilter.FIND_EDGES).load()
    stroke = Image.new(img.mode, img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(stroke)
    fill = (0, 0, 0, 180) if color == 'black' else (255, 255, 255, 180)
    for x in range(X):
        for y in range(Y):
            if edge[x, y][3] > 0:
                draw.ellipse((x - size, y - size, x + size, y + size), fill=fill)
    stroke.paste(img, (0, 0), img)

    return stroke


@torch.no_grad()
def vis_opaque_img_border(data_batch, heatmaps, rf=False, alpha=0.4, vis_th=0.05, crop_th=0.05,
                          kernel_size=39) -> Image.Image:
    """
    Draws reference images. The function lowers the opacity in regions with relevance lower than max(relevance)*vis_th.
    In addition, the reference image can be cropped where relevance is less than max(relevance)*crop_th by setting 'rf' to True.

    Parameters:
    ----------
    data_batch: torch.Tensor
        original images from dataset without FeatureVisualization.preprocess() applied to it
    heatmaps: torch.Tensor
        ouput heatmap tensor of the CondAttribution call
    rf: boolean
        Computes the CRP heatmap for a single neuron and hence restricts the heatmap to the receptive field.
        The amount of cropping is further specified by the 'crop_th' argument.
    alpha: between [0 and 1]
        Regulates the transparency in low relevance regions.
    vis_th: between [0 and 1)
        Visualization Threshold: Increases transparency in regions where relevance is smaller than max(relevance)*vis_th.
    crop_th: between [0 and 1)
        Cropping Threshold: Crops the image in regions where relevance is smaller than max(relevance)*crop_th.
        Cropping is only applied, if receptive field 'rf' is set to True.
    kernel_size: scalar
        Parameter of the torchvision.transforms.functional.gaussian_blur function used to smooth the CRP heatmap.

    Returns:
    --------
    image: list of PIL.Image objects
        If 'rf' is True, reference images have different shapes.
    """

    if alpha > 1 or alpha < 0:
        raise ValueError("'alpha' must be between [0, 1]")
    if vis_th >= 1 or vis_th < 0:
        raise ValueError("'vis_th' must be between [0, 1)")
    if crop_th >= 1 or crop_th < 0:
        raise ValueError("'crop_th' must be between [0, 1)")

    imgs = []
    for i in range(len(data_batch)):

        img = data_batch[i]

        filtered_heat = gaussian_blur(heatmaps[i].unsqueeze(0), kernel_size=[kernel_size])[0]
        filtered_heat = filtered_heat / (filtered_heat.abs().max())
        vis_mask = filtered_heat > vis_th
        # imgs.append(imgify(img.detach().cpu()).convert('RGB'))
        # continue
        if True:
            row1, row2, col1, col2 = get_crop_range(filtered_heat, crop_th)

            dr = row2 - row1
            dc = col2 - col1
            if dr > dc:
                col1 -= (dr - dc) // 2
                col2 += (dr - dc) // 2
                if col1 < 0:
                    col2 -= col1
                    col1 = 0
            elif dc > dr:
                row1 -= (dc - dr) // 2
                row2 += (dc - dr) // 2
                if row1 < 0:
                    row2 -= row1
                    row1 = 0

            img_t = img[..., row1:row2, col1:col2]
            vis_mask_t = vis_mask[row1:row2, col1:col2]

            if img_t.sum() != 0 and vis_mask_t.sum() != 0:
                # check whether img_t or vis_mask_t is not empty
                img = img_t
                vis_mask = vis_mask_t

        # vis_mask = scipy.ndimage.gaussian_filter(vis_mask.float().cpu().numpy() * 1.0, 4)
        # vis_mask = torch.from_numpy(vis_mask / vis_mask.max()).to(img.device)
        # inv_mask = 1 - vis_mask
        inv_mask = ~vis_mask
        outside = (img * vis_mask).sum((1, 2)).mean(0) / stabilize(vis_mask.sum()) > 0.5

        img = img * vis_mask + img * inv_mask * alpha + outside * 0 * inv_mask * (1 - alpha)

        img = imgify(img.detach().cpu()).convert('RGBA')

        img_ = np.array(img).copy()
        img_[..., 3] = (vis_mask * 255).detach().cpu().numpy().astype(np.uint8)
        img_ = mystroke(Image.fromarray(img_), 0, color='black' if outside else 'black')

        img.paste(img_, (0, 0), img_)

        imgs.append(img.convert('RGB'))

    return imgs


def get_prototypes(attr, embedding_attr, act, embedding_act):
    prototypes = []
    for i, (X, emb) in enumerate([(attr, embedding_attr), (act, embedding_act)]):
        gmm = GaussianMixture(n_components=8, random_state=0).fit(X.detach().cpu().numpy())

        prototypes.append(gmm.means_)
    return prototypes


def get_top_concepts():
    pass


def get_umaps(attr, act, model_dir):
    embedding_attr = UMAP(n_neighbors=5, random_state=123, n_jobs=1)
    X_attr_filename = model_dir + "X_attr.npy"
    X_attr = embedding_attr.fit_transform(attr.detach().cpu().numpy())
    with open(X_attr_filename, 'wb') as f:
        np.save(f, X_attr)

    embedding_act = UMAP(n_neighbors=5, random_state=123, n_jobs=1)
    X_act_filename = model_dir + "X_act.npy"
    if os.path.exists(X_act_filename):
        with open(X_act_filename, 'rb') as f:
            X_act = np.load(X_act_filename)
    else:
        X_act = embedding_attr.fit_transform(act.detach().cpu().numpy())
        with open(X_act_filename, 'wb') as f:
            np.save(f, X_act)

    return embedding_attr, embedding_act, X_attr, X_act

# debug fully loads the dataset but then only uses the first few samples for testing purposes
def cluster_explanations(model, data_dir, model_dir, genes, debug=False):
    for gene in genes:
        print("clustering started for gene", gene)
        model.eval()
        composite, layer_type, layer_name = get_composite_layertype_layername_lightning(model)
        out_path = model_dir + "/crp_out/"

        os.makedirs(out_path, exist_ok=debug)
        print("loading dataset")
        dataset = get_dataset_for_plotting(data_dir, genes=[gene], device=model.device)
        print("dataset loaded")
        if debug:
            dataset.dataframe = dataset.dataframe.drop(list(range(10, len(dataset))))
        attribution = CondAttribution(model)

        print("target layer:", layer_name)
        print("target layer type:", layer_type)
        layer_map = {layer_name: ChannelConcept()}

        fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=None, path=out_path)
        print("preprocessing CRP to ", out_path)
        fv.run(composite, 0, len(dataset) // 1, batch_size=32)
        print("CRP preprocessing done. output path:", out_path)

        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

        cc = ChannelConcept()

        print("calculating concept attributions and activations over the dataset")
        activations, attributions, outputs, indices = calculate_attributions(dataloader, model.device, composite, layer_name, attribution, out_path, cc)
        print("calculation finished")

        # all indices are 0 because we only have one output
        attr = attributions[outputs.argmax(1) == 0]
        act = activations[outputs.argmax(1) == 0]
        indices = indices[outputs.argmax(1) == 0]
        print(attr)

        embedding_attr, embedding_act, X_attr, X_act = get_umaps(attr, act, model_dir)
        x_attr, y_attr = X_attr[:, 0], X_attr[:, 1]
        x_act, y_act = X_act[:, 0], X_act[:, 1]

        print("calculating prototypes")
        os.makedirs(out_path + "/prototypes/", exist_ok=debug)
        prototypes = get_prototypes(attr, embedding_attr, act, embedding_act)

        for i in range(len(prototypes)):
            np.save(out_path + "/prototypes/" + str(i), prototypes[i])

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
                img = dataset[prototype_samples[j][i]][0]
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

        n_refimgs = 12
        print("top concepts:", top_concepts)
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
        result_path = out_path + "/result_" + gene + ".png"
        plt.savefig(result_path)
        plt.clf()
        print("clustering result saved to", result_path)

