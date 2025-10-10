
import zennit
from zennit.attribution import Gradient


import torch
import pandas as pd

import matplotlib.pyplot as plt
import os


def get_img_target_name(loader, device, tile_no):
    image, target, name = loader[tile_no]
    image = image.unsqueeze(0).to(device)
    image = image.float()
    target = torch.tensor(target[0]).to(device)
    return image, target, name


def get_coords_from_name(data_dir, patient, tile_name):
    base_path = data_dir+patient+"/meta_data/"
    merge = pd.read_csv(base_path + "merge.csv", index_col=False)
    line = merge.loc[merge['tile'] == tile_name]
    x = line['x'].to_list()[0]
    y = line['y'].to_list()[0]

    return x, y


def plot_relevance(att, filename=None, only_return=False):
    if filename is None:
        rel = att.sum(1).cpu()
    else:
        rel = torch.tensor(plt.imread(filename)).unsqueeze(0)

    rel = rel / (abs(rel).max()+1e-12)
    # create an image of the visualize attribution
    img = zennit.image.imgify(rel, symmetric=True, cmap='coldnhot', vmin=-1, vmax=1)

    # show the image
    if not only_return:
        display(img)
    return img


def relevance_and_plot(model, composite, device, input=None):
    if input is None:
        input = torch.randn(1, 3, 224, 224).to(device)
    with Gradient(model, composite) as attributor:
        out, grad = attributor(input)
    imshow = input.to('cpu').squeeze().numpy().sum(axis=0)
    plt.imshow(imshow)
    plot_relevance(grad)
    print("out: ", out)


def get_attributions_from_loader(model, loader, device, data_dir, patient, composite, j=-1, iterations=20):
    out_target = []
    if j >= 0:
        iterations = 1
    for i in range(iterations):
        if j >= 0:
            i = j
        input, target, name = get_img_target_name(loader,device,i)
        x, y = get_coords_from_name(data_dir,patient,os.path.basename(name))
        with Gradient(model, composite) as attributor:
            out, grad = attributor(input)
            if grad.count_nonzero() == 0:
                continue
        out_target.append((out, 0, 0, grad, target, x, y, name))
    return out_target
