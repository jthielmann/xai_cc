import torch
import os
import faulthandler; faulthandler.enable()
import torchmetrics

import pandas as pd
from model import load_model
import json
from data_loader import get_patient_loader, get_train_samples, get_val_samples, get_test_samples
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
data_dir_train = "../Training_Data/"
data_dir_test = "../Test_Data/"
patients_train = get_train_samples()
patients_val = get_val_samples()
patients_test = get_test_samples()
patients_test = [patients_test[0], patients_test[2]]
print(patients_train)
print(patients_val)
print(patients_test)

model_dir = "../models/"
model_dir_path = []
# gather new models only

model_list_file_name = "new_models.csv"
update_model_list = True


def generate_model_list(model_dir):
    for model_type_dir in os.listdir(model_dir):
        print(model_type_dir)
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

                # skip old models
                if "genes" not in d:
                    continue

            files = os.listdir(sub_path)
            for f in files:
                if os.path.basename(f) == "best_model.pt":
                    model_dir_path.append((sub_path + "/", sub_path + "/" + f))
    frame = pd.DataFrame(model_dir_path, columns=["model_dir", "model_path"])
    frame.to_csv(model_dir + model_list_file_name, index=False)
    return frame


if not os.path.exists(model_dir + model_list_file_name) or update_model_list:
    frame = generate_model_list(model_dir)
else:
    frame = pd.read_csv(model_dir + model_list_file_name)


def generate_results(model_frame, patients, data_dir, out_file_appendix=""):
    for idx, row in model_frame.iterrows():
        results_filename = row["model_dir"] + os.path.basename(row["model_path"][:-3]) + out_file_appendix + "_results.csv"
        """    if results_filename.find("resnet50/MKI67_random/") == -1:
            continue"""
        print(results_filename)

        token_name = row["model_dir"] + "generation_token" + out_file_appendix
        if os.path.exists(results_filename) and not os.path.exists(token_name):
            os.remove(results_filename)
        if os.path.exists(token_name):
            continue

        open(token_name, "a").close()
        model = load_model(row["model_dir"], row["model_path"], squelch=True).to(device).eval()

        columns_base = ["path"]
        for gene in model.gene_list:
            columns_base.append("labels_" + gene)
        for gene in model.gene_list:
            columns_base.append("out_" + gene)

        for patient in patients:
            columns = columns_base.copy()
            columns.append("patient")

            with torch.no_grad():
                df = pd.DataFrame(columns=columns)
                loader = get_patient_loader(data_dir, patient, model.gene_list)
                for img, labels, path in loader:
                    out = model(img.unsqueeze(0).to(device))
                    out_row = [path]
                    for label in labels:
                        out_row.append(label)
                    for out_item in out:
                        if out_item.shape == torch.Size([1]):
                            out_row.append(out_item.item())
                        else:
                            for t in out_item:
                                out_row.append(t.item())
                    out_row.append(patient)
                    #print(columns_base, columns, out_row)
                    df.loc[len(df)] = out_row
                if not os.path.isfile(results_filename):
                    df.to_csv(results_filename, header=columns)
                else: # else it exists so append without writing the header
                    df.to_csv(results_filename, mode='a', header=False)


generate_results(frame, patients_train, data_dir_train, "_train")
generate_results(frame, patients_val, data_dir_train, "_val")
#generate_results(frame, patients_test, data_dir_test, "_test")


print("done")
