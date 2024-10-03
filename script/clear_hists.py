import torch
import os

import torchmetrics

import pandas as pd
from model import load_model
import json
from data_loader import get_patient_loader
import matplotlib.pyplot as plt
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

for idx, row in frame.iterrows():
    results_filename = row["model_dir"] + "/" + os.path.basename(row["model_path"][:-3]) + "_results.csv"
    """    if os.path.exists(results_filename):
        print("removing", results_filename)
        os.remove(results_filename)
    token_name = row["model_dir"] + "generation_token"
    if os.path.exists(token_name):
        print("removing", token_name)
        os.remove(token_name)"""
    model = load_model(row["model_dir"], row["model_path"], squelch=True).to(device).eval()
    scatter_name = row["model_dir"] + "scatter.png"
    if os.path.exists(scatter_name):
        print("removing", scatter_name)
        os.remove(scatter_name)
    hist_name = row["model_dir"] + "hist_token"
    if os.path.exists(hist_name):
        print("removing", hist_name)
        os.remove(hist_name)
    for gene in model.gene_list:
        scatter_name = row["model_dir"] + gene + "_scatter.png"
        if os.path.exists(scatter_name):
            print("removing", scatter_name)
            os.remove(scatter_name)


