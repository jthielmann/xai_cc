import torch
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
from model import load_model

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
data_dir_train = "../Training_Data/"
data_dir_test = "../Test_Data/"
patients_train = [os.path.basename(f) for f in os.scandir(data_dir_train) if f.is_dir()]
patients_test = [os.path.basename(f) for f in os.scandir(data_dir_test) if f.is_dir()][1:]
print(patients_train)
print(patients_test)

model_dir = "../models/"
skip = 0
model_dir_path = []
# gather new models only

model_list_file_name = "new_models.csv"
update_model_list = False
if not os.path.exists(model_dir + model_list_file_name) or update_model_list:
    for model_type_dir in os.listdir(model_dir):
        sub_path = model_dir + model_type_dir
        if model_type_dir == ".DS_Store" or model_type_dir == "new" or os.path.isfile(sub_path):
            continue
        for model_leaf_dir in os.listdir(sub_path):
            sub_path = model_dir + model_type_dir + "/" + model_leaf_dir
            if model_type_dir == ".DS_Store" or os.path.isfile(sub_path):
                continue

            with open(sub_path + "/settings.json") as settings_json:
                d = json.load(settings_json)
                model_type = d["model_type"]

                if "genes" not in d:
                    continue

            files = os.listdir(sub_path)
            for f in files:
                if f[-3:] == ".pt" and f.find("ep_") != -1:
                    src = sub_path + "/" + f
                    dst = sub_path + "/" + f[f.find("ep_"):]
                    os.rename(src, dst)
                    if f[f.find("ep_"):] == "ep_29.pt":
                        model_dir_path.append((sub_path + "/", dst))

    frame = pd.DataFrame(model_dir_path, columns=["model_dir", "model_path"])
    frame.to_csv(model_dir + model_list_file_name, index=False)
else:
    frame = pd.read_csv(model_dir + model_list_file_name)

"""row = frame.iloc[0]
img = plt.imread(row["model_dir"] + "/scatter.png")
imgplot = plt.imshow(img)

plt.axis('off')
plt.show()"""


def plot_hist_comparison(img_paths, width=4, subplot_size=10, gene=None):

    height = int((len(img_paths)) / width + 0.999)
    f, ax = plt.subplots(height,width, figsize=(50,50), dpi=300)
    f.set_figheight(subplot_size * int(height / width + 0.999))
    f.set_figwidth(subplot_size)
    for i in range(len(img_paths)):
        if not os.path.exists(img_paths[i]):
            print("missing: ", img_paths[i])
            continue
        img = plt.imread(img_paths[i])
        x = int(i/width)
        y = i%width
        if height < 2:
            ax[y].imshow(img)
            ax[y].axis('off')
        else:
            ax[x, y].imshow(img)
            ax[x, y].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    if gene:
        plt.savefig("../" + gene + "_hists.png")
    else:
        plt.savefig("../hists.png")

    plt.clf()


path = "../models/resnet50/VWF_random_freeze/scatter.png"
#array = [path, path, path, path,path]
#plot_hist_comparison(array)
genes = []
image_paths = []
for _, row in frame.iterrows():
    model_dir = row["model_dir"]
    model_path = row["model_path"]
    model = load_model(model_dir, model_path, squelch=True)
    for gene in model.gene_list:
        file_name = model_dir + "/" + gene + "_scatter.png"
        try:
            gene_idx = genes.index(gene)
        except ValueError:
            gene_idx = -1
        if gene_idx < 0:
            genes.append(gene)
            image_paths.append([file_name])
        else:
            image_paths[gene_idx].append(file_name)

for i in range(len(genes)):
    plot_hist_comparison(image_paths[i], gene=genes[i])



"""for path in frame["model_dir"] + "/scatter.png":
    print(path)
    continue
    plt.imshow(plt.imread(path))
    plt.show()
"""

