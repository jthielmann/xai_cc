import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import numpy as np
from pathlib import Path


def read_meta_data(patients, dataset_name, data_dir, meta_data_dir, csv_name, gene):
    df = None
    for patient in patients:
        file_name = data_dir + "/" + dataset_name + "/" + patient + "/" + meta_data_dir + "/" + csv_name
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

    all_stats = []
    for gene in genes:
        train_df = read_meta_data(train_patients, dataset_name, data_dir, meta_data_dir, csv_name, gene)
        val_df = read_meta_data(val_patients, dataset_name, data_dir, meta_data_dir, csv_name, gene)
        test_df = read_meta_data(test_patients, dataset_name, data_dir, meta_data_dir, csv_name, gene)

        # TODO: generate stats to compare such as mean std ... that are helpful to check if the test split is decent and representative
        stats = {"gene": gene}
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
        all_stats.append(stats)

    # TODO: save file to save_dir + dataset_name + ".csv"
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(save_dir, f"{dataset_name}.csv")
        stats_df.to_csv(file_path, index=False)
        print(f"Saved stats to {file_path}")


if __name__ == "__main__":


    data_dir = "/Users/thielmann/Documents/xai_cc/data/"
    dataset_name = "crc_base"
    train_patients = ["Training_Data/p007", "Training_Data/p013", "Training_Data/p014"]
    val_patients = ["Training_Data/p009"]
    test_patients = ["Training_Data/p020"]
    genes = ["RUBCNL"]
    save_dir = "/Users/thielmann/Documents/code/xai_cc/docs/gene_dist_comparison/"
    meta_data_dir_name = "meta_data"
    csv_name = "gene_data.csv"

    evaluate_split_dists(data_dir, dataset_name, train_patients, val_patients, test_patients, genes, save_dir, meta_data_dir_name, csv_name)
