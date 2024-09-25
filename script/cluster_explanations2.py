import numpy as np
import matplotlib.pyplot as plt
import torch
import zennit
from IPython.core.display_functions import display
from zennit.torchvision import VGGCanonizer, ResNetCanonizer
from zennit.composites import EpsilonPlusFlat
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
from model import get_remote_models_and_path
from tqdm import tqdm
from torchvision.utils import make_grid
import zennit.image as zimage
from crp.image import imgify




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

cc = ChannelConcept()

activations = []
attributions = []
outputs = []

#
if not already_calculated or not os.path.exists(out_path + "/activations.pt"):
    print("calculating concept attributions and activations over the dataset")

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
else:
    print("loading concept attributions and activations over the dataset")
    activations = torch.load(out_path + "/activations.pt")
    attributions = torch.load(out_path + "/attributions.pt")
    outputs = torch.load(out_path + "/outputs.pt")
    indices = torch.load(out_path + "/indices.pt")
    print(activations.shape, attributions.shape, outputs.shape, indices.shape)


# all indices are 0 because we only have one output
attr = attributions[outputs.argmax(1) == 0]
act = activations[outputs.argmax(1) == 0]
indices = indices[outputs.argmax(1) == 0]

from umap import UMAP

embedding_attr = UMAP(n_neighbors=5, random_state=123, n_jobs=1)
X_attr = embedding_attr.fit_transform(attr.detach().cpu().numpy())
x_attr, y_attr = X_attr[:, 0], X_attr[:, 1]

embedding_act = UMAP(n_neighbors=5, random_state=123, n_jobs=1)
X_act = embedding_act.fit_transform(act.detach().cpu().numpy())
x_act, y_act = X_act[:, 0], X_act[:, 1]

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
plt.savefig(out_path + "/umap.png")
fig.show()

from sklearn.mixture import GaussianMixture

prototypes = []
for i, (X, emb) in enumerate([(attr, embedding_attr), (act, embedding_act)]):
    gmm = GaussianMixture(n_components=8, random_state=0).fit(X.detach().cpu().numpy())
    prototypes.append(gmm.means_)
    mean = emb.transform(gmm.means_)
    axes[i].scatter(mean[:, 0], mean[:, 1], s=30, c="#1B4365", zorder=2, label="prototypes")
    for k, prot in enumerate(mean):
        axes[i].text(prot[0], prot[1], k, fontsize=4, color="white", ha="center", va="center")
    axes[i].legend()

display(fig)

proto_attr = prototypes[0]

distances = np.linalg.norm(attr[:, None, :].detach().cpu() - proto_attr, axis=2)
prototype_samples = np.argsort(distances, axis=0)[:8]
prototype_samples = indices[prototype_samples]

fig, axs = plt.subplots(1, 8, figsize=(6, 8), dpi=200, facecolor='white')


N_PROTOTYPES = 8
for i in range(N_PROTOTYPES):
    grid = make_grid(
        [dataset[prototype_samples[j][i]][0] for j in range(8)],
        nrow=1,
        padding=0)
    grid = np.array(zimage.imgify(grid.detach().cpu()))
    img = imgify(grid)
    axs[i].imshow(img)
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_title(f"{i}")


import numpy as np
import scipy.ndimage
import torch
from PIL import ImageFilter, Image, ImageDraw
from crp.image import get_crop_range, imgify
from torchvision.transforms.functional import gaussian_blur
from zennit.core import stabilize


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

        filtered_heat = gaussian_blur(heatmaps[i].unsqueeze(0), kernel_size=kernel_size)[0]
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

import torchvision

proto = torch.from_numpy(proto_attr)
top_concepts = torch.topk(proto, 3).indices.flatten().unique()
top_concepts = top_concepts[proto[:, top_concepts].amax(0).argsort(descending=True)].tolist()
concept_matrix = proto[:, top_concepts].T
N_CONCEPTS = len(top_concepts)

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


