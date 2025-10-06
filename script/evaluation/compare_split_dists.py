import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import numpy as np
from pathlib import Path

def read_meta_data(patients, dataset_name, data_dir, meta_data_dir, csv_name, gene):
    once = True
    df = None
    for patient in patients:
        file_name = data_dir + "/" + dataset_name + "/" + patient + "/" + meta_data_dir + "/" + csv_name
        if not once:
            print(file_name)
            once = True
        if not os.path.exists(file_name):
            continue

        patient_df = pd.read_csv(file_name, usecols=[gene])
        patient_df["patient"] = patient
        if df is None:
            df = patient_df
        else:
            df = pd.concat([df, patient_df])
    return df

def evaluate_split_dists(data_dir, dataset_name, train_patients, val_patients, test_patients, genes, save_dir, meta_data_dir, csv_name):

    for gene in genes:
        train_df = read_meta_data(train_patients, dataset_name, data_dir, meta_data_dir, csv_name, gene)
        val_df = read_meta_data(val_patients, dataset_name, data_dir, meta_data_dir, csv_name, gene)
        test_df = read_meta_data(test_patients, dataset_name, data_dir, meta_data_dir, csv_name, gene)

        # TODO: generate stats to compare such as mean std ... that are helpful to check if the test split is decent and representative
        stats = {}
        for split_name, df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
            if df is not None and not df.empty:
                stats[f'{split_name}_mean'] = df[gene].mean()
                stats[f'{split_name}_std'] = df[gene].std()
                stats[f'{split_name}_count'] = len(df)

        if train_df is not None and not train_df.empty and test_df is not None and not test_df.empty:
            ks_stat, p_value = ks_2samp(train_df[gene], test_df[gene])
            stats['ks_stat_train_test'] = ks_stat
            stats['p_value_train_test'] = p_value

        if train_df is not None and not train_df.empty and val_df is not None and not val_df.empty:
            ks_stat, p_value = ks_2samp(train_df[gene], val_df[gene])
            stats['ks_stat_train_val'] = ks_stat
            stats['p_value_train_val'] = p_value

        print(stats)

        # TODO: save file to save_dir + dataset_name + ".csv"


if __name__ == "__main__":


    data_dir = "/data/cephfs-2/unmirrored/groups/krieger/xai/HEST/"
    dataset_name = "hest_coad_visium"
    #train_patients = ["TENX92","TENX91","TENX90","TENX89","TENX70","TENX49",
    #    "ZEN49","ZEN48","ZEN47","ZEN46","ZEN45","ZEN44"]
    #val_patients = ["TENX29","ZEN43","ZEN42","ZEN40","ZEN39","ZEN38","ZEN36"]
    #test_patients = []

    train_patients = [
         'MISC62', 'ZEN45', 'TENX152', 'MISC73', 'MISC71', 'MISC70', 'MISC68', 'MISC67', 'MISC66', 'MISC65', 'MISC64',
         'MISC63', 'MISC58', 'MISC57', 'MISC51', 'MISC50', 'MISC49', 'MISC48', 'MISC47', 'MISC46', 'MISC44', 'MISC43',
         'MISC41', 'MISC40', 'MISC39', 'MISC38', 'MISC36', 'TENX92', 'TENX91', 'TENX49', 'ZEN47', 'ZEN46', 'ZEN43',
         'ZEN39', 'MISC42']
    val_patients = ['MISC69', 'MISC56', 'MISC37', 'MISC35', 'MISC34', 'TENX90', 'TENX29', 'ZEN42', 'ZEN38']
    test_patients = ['MISC72', 'MISC45', 'MISC33', 'TENX89', 'TENX28', 'ZEN44']


    genes = ["RUBCNL"]
    save_dir = "../docs/gene_dist_comparison/"
    meta_data_dir_name = "metadata"
    csv_name = "gene_log1p.csv"

    evaluate_split_dists(data_dir, dataset_name, train_patients, val_patients, test_patients, genes, save_dir, meta_data_dir_name, csv_name)
