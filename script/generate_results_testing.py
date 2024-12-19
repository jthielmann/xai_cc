import pandas
import torch
import os
import faulthandler; faulthandler.enable()

import pandas as pd
from model import load_model, generate_model_list
from data_loader import get_patient_loader, get_train_samples, get_val_samples, get_test_samples
import torchmetrics
import scipy

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
patients_train = get_train_samples()
patients_val = get_val_samples()
samples_crc_n19 = [["TENX92","TENX91","TENX90","TENX89","TENX70","TENX49", "ZEN49", "ZEN48", "ZEN47", "ZEN46", "ZEN45", "ZEN44"],
           ["TENX29", "ZEN43", "ZEN42", "ZEN40", "ZEN39", "ZEN38", "ZEN36"]]

data_dirs_train_base = []
data_dirs_val_base = []

data_dirs_train_n19 = []
data_dirs_val_n19 = []

data_dirs_train_base.append(("../data/jonas/Training_Data/", patients_train))
data_dirs_train_n19.append(("../data/CRC-N19/", samples_crc_n19[0]))

data_dirs_val_base.append(("../data/jonas/Training_Data/", patients_val))
data_dirs_val_n19.append(("../data/CRC-N19/", samples_crc_n19[1]))

patients_test = get_test_samples()
patients_test = [patients_test[0], patients_test[2]]
print(patients_train)
print(patients_val)
print(patients_test)

model_dir = "../models_2/"
model_dir_path = []
# gather new models only

model_list_file_name = "new_models.csv"
update_model_list = True

must_contain = None
skip_names = ["AE", "dropout"]
"""
frame = pandas.DataFrame(columns=["model_dir", "model_path"])
frame.loc[0] = ["../models/resnet18/RUBCNL/", "../models/resnet18/RUBCNL/ep_29.pt"]
"""

if not os.path.exists(model_dir + model_list_file_name) or update_model_list:
    print("found these models:")
    frame = generate_model_list(model_dir, must_contain=must_contain, skip_names=skip_names)
else:
    frame = pd.read_csv(model_dir + model_list_file_name)

for idx, row in frame.iterrows():
    print(idx, row["model_path"])

def generate_results(model_frame, data_dirs, mode="train", calculate_anyway=False):
    for idx, row in model_frame.iterrows():
        results_filename = row["model_dir"] + os.path.basename(row["model_path"][:-3]) + "_" + mode + "_results.csv"

        print(results_filename)

        token_name = row["model_dir"] + "generation_token9_" + mode
        if os.path.exists(results_filename) and not os.path.exists(token_name):
            os.remove(results_filename)
        if os.path.exists(token_name) and not calculate_anyway:
            print("already calculated, continuing")
            continue
        if calculate_anyway:
            if os.path.exists(results_filename):
                os.remove(results_filename)

        open(token_name, "a").close()
        model = load_model(row["model_dir"], row["model_path"], squelch=False)
        if model is None:
            continue
        model = model.to(device)
        model.eval()
        columns = ["path"]
        for gene in model.gene_list:
            columns.append("labels_" + gene)
        for gene in model.gene_list:
            columns.append("out_" + gene)
        columns.append("patient")
        for data_dir, samples in data_dirs:

            for patient in samples:
                try:

                    loader = get_patient_loader(data_dir, patient, model.gene_list)
                    print("using", data_dir, patient)
                except :
                    continue

                with torch.no_grad():
                    df = pd.DataFrame(columns=columns)

                    if loader is None:
                        continue
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
                        df.loc[len(df)] = out_row
                    if not os.path.isfile(results_filename):
                        df.to_csv(results_filename, header=columns)
                    else: # else it exists so append without writing the header
                        df.to_csv(results_filename, mode='a', header=False)
        if not os.path.exists(results_filename):
            continue
        df = pd.read_csv(results_filename)
        print(df.head())
        columns = []
        for gene in model.gene_list:
            columns.append((gene, "labels_" + gene, "out_" + gene))

        for gene, column_target, column_out in columns:
            mse = torchmetrics.MeanSquaredError()

            out = torch.tensor(df[column_out].to_numpy())
            target = torch.tensor(df[column_target].to_numpy())
            mse_value = round(mse(out, target).item(), 3)
            pearson = round(scipy.stats.pearsonr(out, target)[0], 3)

            print(gene, mse_value, pearson)


print("----------------------------------------------------")
print("starting results generation")

calculate_anyway = False
generate_results(frame, data_dirs_train_n19, mode = "train_n19", calculate_anyway=calculate_anyway)
generate_results(frame, data_dirs_val_n19, mode = "val_n19", calculate_anyway=calculate_anyway)

print("generate_results done")
print("--------------------------------------------------")
out_filename = "../results/results_base.csv"
out_filename_mean = "../results/results_mean_base.csv"


output_appendix_train = "_train_base"
output_appendix_val = "_val_base"
for idx, row in frame.iterrows():
    print(row["model_path"])
    results_filename_train = row["model_dir"] + os.path.basename(row["model_path"][:-3]) + output_appendix_train + "_results.csv"
    results_filename_val = row["model_dir"] + os.path.basename(row["model_path"][:-3]) + output_appendix_val + "_results.csv"
    if os.path.exists(results_filename_train):
        train_result_df = pd.read_csv(results_filename_train)
    else:
        print("train results not found:", results_filename_train)
        continue
    if os.path.exists(results_filename_val):
        val_result_df = pd.read_csv(results_filename_val)
    else:
        print("val results not found:", results_filename_val)
        continue

    model = load_model(row["model_dir"], row["model_path"], squelch=True)
    if model is None:
        continue
    model = model.to(device).eval()

    columns = []
    for gene in model.gene_list:
        columns.append((gene, "labels_" + gene, "out_" + gene))


    row_for_csv = []

    for gene, column_label, column_out in columns:
        mse = torchmetrics.MeanSquaredError()

        out_train = torch.tensor(train_result_df[column_out].to_numpy())
        label_train = torch.tensor(train_result_df[column_label].to_numpy())
        mse_train = round(mse(out_train, label_train).item(), 3)
        pearson_train = round(scipy.stats.pearsonr(out_train, label_train)[0], 3)

        out_val = torch.tensor(val_result_df[column_out].to_numpy())
        label_val = torch.tensor(val_result_df[column_label].to_numpy())
        mse_val = round(mse(out_val, label_val).item(),3)
        pearson_val = round(scipy.stats.pearsonr(out_val, label_val)[0], 3)

        row_for_csv.append((gene, mse_train, pearson_train, mse_val, pearson_val))

    results = []
    if os.path.exists(out_filename):
        results_df = pd.read_csv(out_filename, index_col="model_path")
    else:
        results_df = pd.DataFrame(columns=["model_path"])
        results_df.set_index("model_path", drop=True, inplace=True)

    for gene, mse_train, pearson_train, mse_val, pearson_val in row_for_csv:
        results_df.at[row["model_path"], gene + "_mse_train"] = mse_train
        results_df.at[row["model_path"], gene + "_pearson_train"] = pearson_train
        results_df.at[row["model_path"], gene + "_mse_val"] = mse_val
        results_df.at[row["model_path"], gene + "_pearson_val"] = pearson_val
    results_df.to_csv(out_filename)

    if os.path.exists(out_filename_mean):
        results_df_mean = pd.read_csv(out_filename_mean, index_col="model_path")
    else:
        results_df_mean = pd.DataFrame(columns=["model_path"])
        results_df_mean.set_index("model_path", drop=True, inplace=True)

    mse_values_train = []
    mse_values_val = []
    pearson_values_train = []
    pearson_values_val = []

    for gene, mse_train, pearson_train, mse_val, pearson_val in row_for_csv:
        mse_values_train.append(mse_train)
        mse_values_val.append(mse_val)
        pearson_values_train.append(pearson_train)
        pearson_values_val.append(pearson_val)

    mse_train_mean = round(sum(mse_values_train) / len(mse_values_train), 3)
    mse_val_mean = round(sum(mse_values_val) / len(mse_values_val), 3)
    pearson_train_mean = round(sum(pearson_values_train) / len(pearson_values_train), 3)
    pearson_val_mean = round(sum(pearson_values_val) / len(pearson_values_val), 3)

    results_df_mean.at[row["model_path"], "mse_train_mean"] = mse_train_mean
    results_df_mean.at[row["model_path"], "mse_val_mean"] = mse_val_mean
    results_df_mean.at[row["model_path"], "pearson_train_mean"] = pearson_train_mean
    results_df_mean.at[row["model_path"], "pearson_val_mean"] = pearson_val_mean
    results_df_mean.to_csv(out_filename_mean)

print("done")
