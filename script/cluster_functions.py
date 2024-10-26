import os.path

import numpy as np
import scipy.ndimage
import torch
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


def get_composite_layertype_layername(model):
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
    return composite, layer_type, layer_name

def calculate_attributions(dataset, device, composite, layer_name, attribution, out_path, cc):
    activations = []
    attributions = []
    outputs = []
    for i, (x, _) in enumerate(tqdm(dataset)):
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
    indices = np.arange(len(dataset))
    torch.save(indices, out_path + "/indices.pt")
    return activations, attributions, outputs, indices


def load_attributions(out_path):
    activations = torch.load(out_path + "/activations.pt", weights_only=False)
    attributions = torch.load(out_path + "/attributions.pt", weights_only=False)
    outputs = torch.load(out_path + "/outputs.pt", weights_only=False)
    indices = torch.load(out_path + "/indices.pt", weights_only=False)
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
    if os.path.exists(X_attr_filename):
        with open(X_attr_filename, 'rb') as f:
            X_attr = np.load(X_attr_filename)
    else:
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




