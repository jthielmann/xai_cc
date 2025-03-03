import scipy
import torch
import os

import torchmetrics

import pandas as pd
from script.model.model import load_model
import json
import matplotlib.pyplot as plt

def update_model_list(model_dir, model_list_file_name = "new_models.csv"):
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

                # skip old models
                if "genes" not in d:
                    continue

            files = os.listdir(sub_path)
            for f in files:
                model_name = "best_model.pt"
                if f.find(model_name) != -1:
                    model_dir_path.append((sub_path + "/", sub_path + "/" + model_name))

    frame = pd.DataFrame(model_dir_path, columns=["model_dir", "model_path"])
    frame.to_csv(model_dir + model_list_file_name, index=False)
    return frame


def generate_hists_loop(frame, out_file_appendix=""):
    for idx, row in frame.iterrows():
        results_filename = row["model_dir"] + os.path.basename(row["model_path"][:-3]) + out_file_appendix + "_results.csv"
        print(results_filename)
        token_name = row["model_dir"] + "hist_token" + out_file_appendix
        if not os.path.exists(results_filename):
            print(results_filename, "not found")
            continue
        if os.path.exists(token_name):
            continue

        open(token_name, "a").close()
        model = load_model(row["model_dir"], row["model_path"], squelch=True).to(device).eval()

        results_file = pd.read_csv(results_filename)
        patients = results_file[results_file.columns[-1]].unique()
        print(patients)
        for gene in model.gene_list:
            out_string = "out_" + gene
            labels_string = "labels_" + gene
            for patient in patients:
                patient_df = results_file[results_file[results_file.columns[-1]] == patient]
                plt.scatter(patient_df[out_string], patient_df[labels_string], label=patient)
            mse = torchmetrics.MeanSquaredError()
            out = torch.tensor(results_file[out_string].to_numpy())
            label = torch.tensor(results_file[labels_string].to_numpy())
            result = mse(out, label)
            pearson = round(scipy.stats.pearsonr(out, label)[0], 2)

            plt.text(x=-2, y=3, s="MSE: " + str(round(result.item(), 2)))
            plt.text(x=-2, y=1, s="pearson: " + str(pearson))
            plt.plot([-2,3],[-2,3], color='red')
            plt.title(results_filename + "\n" + gene)
            plt.legend(loc="lower right")
            plt.xlabel('output')
            plt.ylabel('target')
            plt.savefig(row["model_dir"] + "/" + gene + out_file_appendix + "_scatter.png")
            plt.clf()


def generate_hists(model, model_dir, results_filename, out_file_appendix=""):
    model.eval()
    results_file = pd.read_csv(results_filename)

    patients = results_file[results_file.columns[-1]].unique()
    print(patients)
    figure_paths = []
    for gene in model.genes:
        out_string = "out_" + gene
        labels_string = "labels_" + gene
        for patient in patients:
            patient_df = results_file[results_file[results_file.columns[-1]] == patient]
            plt.scatter(patient_df[out_string], patient_df[labels_string], label=patient)
        mse = torchmetrics.MeanSquaredError()
        out = torch.tensor(results_file[out_string].to_numpy())
        label = torch.tensor(results_file[labels_string].to_numpy())
        result = mse(out, label)
        pearson = round(scipy.stats.pearsonr(out, label)[0], 2)
        print("pearson", scipy.stats.pearsonr(out, label)[0])

        plt.text(x=-2, y=3, s="MSE: " + str(round(result.item(), 2)))
        plt.text(x=-2, y=1, s="pearson: " + str(pearson))
        plt.plot([-2, 3], [-2, 3], color='red')
        plt.title(results_filename + "\n" + gene + out_file_appendix)
        plt.legend(loc="lower right")
        plt.xlabel('output')
        plt.ylabel('target')
        figure_path = model_dir + "/" + gene + out_file_appendix + "_scatter.png"
        figure_paths.append(figure_path)
        plt.savefig(figure_path)
        plt.clf()
    return figure_paths


def plot_hist_comparison(img_paths, width=4, subplot_size=10, gene=None, appendix=""):
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
        plt.savefig("../" + gene + appendix + "_hists.png")
    else:
        plt.savefig("../hists" + appendix + ".png")

    plt.clf()


def gather_genes_and_paths(frame):
    genes = []
    image_paths_train = []
    image_paths_val = []
    for _, row in frame.iterrows():
        model_dir = row["model_dir"]
        model_path = row["model_path"]
        model = load_model(model_dir, model_path, squelch=True)
        print(model_dir)
        for gene in model.gene_list:
            file_name_train = model_dir + "/" + gene + "_train_scatter.png"
            file_name_val = model_dir + "/" + gene + "_val_scatter.png"


            try:
                gene_idx = genes.index(gene)
            except ValueError:
                gene_idx = -1
            if gene_idx < 0:
                genes.append(gene)
                image_paths_train.append([file_name_train])
                image_paths_val.append([file_name_val])
            else:
                image_paths_train[gene_idx].append(file_name_train)
                image_paths_val[gene_idx].append(file_name_val)
    return genes, image_paths_train, image_paths_val


def generate_tile_maps(data_dir, patients, model_info, results_filename):
    column_names = ["tile", "out", "label", "x", "y"]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # we load each model and their results csv
    for model_dir, model_path in model_info:
        print(model_dir)
        if not os.path.exists(model_path) or model_path.find("dino") == -1:
            continue
        model = load_model(model_dir, model_path, squelch=True).to(device)
        # gather results
        results_path = model_dir + "/" + results_filename
        if not os.path.exists(results_path):
            continue
        results = pd.read_csv(results_path, index_col=0)
        # can come with path only, we also fix it here for later occasions
        if "tile" not in results.columns:
            results["tile"] = results["path"].apply(os.path.basename)
            results.to_csv(results_path)
        out_put_dir = model_dir + "/maps"
        if not os.path.exists(out_put_dir):
            os.mkdir(out_put_dir)

        # gather each patient's tile coordinates
        for patient in patients:
            coords = pd.read_csv(data_dir + "/" + patient + "/meta_data/spatial_data.csv")
            for gene in model.gene_list:
                plotting_data = None
                file_name = out_put_dir + "/" + gene + "_" + patient + "_results_with_coords.csv"
                if os.path.exists(file_name):
                    plotting_data = pd.read_csv(file_name)
                else:
                    for _, coord_row in coords.iterrows():
                        tilename = coord_row["tile"]
                        results_row = results.loc[results['tile'] == tilename]
                        if results_row.empty:
                            continue
                        data = [tilename, results_row["out_" + gene], results_row["labels_" + gene], coord_row["x"], coord_row["y"]]
                        row_dict = dict((column_names[i], data[i]) for i in range(len(column_names)))
                        if plotting_data is None:
                            plotting_data = pd.DataFrame(row_dict, columns=column_names)
                        else:
                            plotting_data = pd.concat([plotting_data, pd.DataFrame(row_dict)], ignore_index=True)

                    plotting_data.to_csv(file_name)

                print(gene, patient, model_dir)
                plotting_data["diff"] = plotting_data["label"] - plotting_data["out"]
                plt.scatter(plotting_data['x'], plotting_data['y'], c=plotting_data["diff"], cmap='viridis')
                plt.colorbar(label='labels')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title(model_dir + "\ndiff for patient " + patient + " for gene " + gene)
                plt.savefig(out_put_dir + "/" + gene + "_" + patient + "_diff.png")
                plt.clf()


"""
data_dir_train = "../data/jonas/Training_Data/"
data_dir_test = "../data/jonas/Test_Data/"
patients_train = get_train_samples()
patients_val = get_val_samples()
print(patients_train)
print(patients_val)



model_dir = "../models/"
skip = 0
model_dir_path = []
# gather new models only

model_list_file_name = "new_models.csv"


if not os.path.exists(model_dir + model_list_file_name) or update_model_list:
    frame = update_model_list(model_dir, model_list_file_name)
else:
    frame = pd.read_csv(model_dir + model_list_file_name)

generate_tile_maps(data_dir_train, patients_train, zip(frame["model_dir"].tolist(), frame["model_path"].tolist()), "best_model_train_results.csv")
generate_tile_maps(data_dir_train, patients_val, zip(frame["model_dir"].tolist(), frame["model_path"].tolist()), "best_model_val_results.csv")
exit(0)

genes, image_paths_train, image_paths_val = gather_genes_and_paths(frame)

for i in range(len(genes)):
    plot_hist_comparison(image_paths_train[i], gene=genes[i], appendix="_train")
    plot_hist_comparison(image_paths_val[i], gene=genes[i], appendix="_val")


generate_hists(frame, "_train")
generate_hists(frame, "_val")

generate_maps()
generate_heatmaps()


print("done")
"""