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
model_dir = "../models/"
skip = 0
model_dir_path = []
# gather new models only

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

            with open(sub_path + "/settings.json") as settings_json:
                d = json.load(settings_json)
                model_type = d["model_type"]

                if "genes" not in d:
                    continue

            files = os.listdir(sub_path)
            #print(files)
            if "ep_9.pt" not in files and "settings.json" in files:
                print("rm -rf", sub_path)
                os.rmdir(sub_path)
                continue
            token_name = "generation_token_train"
            if "ep_29_train_results.csv" not in files and token_name in files:
                print("removing", sub_path + "/" + token_name)
                os.remove(sub_path + "/" + token_name)
            token_name = "generation_token_val"
            if "ep_29_val_results.csv" not in files and token_name in files:
                print("removing", sub_path + "/" + token_name)
                os.remove(sub_path + "/" + token_name)
