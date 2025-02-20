from script.data_processing.data_loader import get_base_dataset, get_dataset
import matplotlib.pyplot as plt
import numpy as np
import os
gene = "RUBCNL"
use_base = True
from script.data_processing.image_transforms import get_transforms
use_val = True
use_test = False
def get_mean_std_features(data_dirs, patients, genes = None):
    if genes is None:
        genes = ["RUBCNL"]
    r_sum, g_sum, b_sum = 0, 0, 0
    r_sq_sum, g_sq_sum, b_sq_sum = 0, 0, 0
    num_pixels = 0  # Total number of pixels across all images

    for patient in patients:
        for data_dir in data_dirs:
            if not os.path.exists(data_dir + "/" + patient):
                if data_dirs.index(data_dir) == len(data_dirs) - 1:
                    out_text = "patient" + patient + "does not exist in any datadir, datadirs:"
                    for d in data_dirs:
                        out_text += d
                        out_text += ", "
                    print(out_text)
                continue
            dataset = get_dataset(data_dir, genes, samples=[patient], transforms=get_transforms())
            print("patient", patient)

            for data, target in dataset:
                # Assuming data shape is (3, H, W)
                num_pixels += data.shape[1] * data.shape[2]  # H * W

                r_sum += data[0].sum()
                g_sum += data[1].sum()
                b_sum += data[2].sum()

                r_sq_sum += (data[0] ** 2).sum()
                g_sq_sum += (data[1] ** 2).sum()
                b_sq_sum += (data[2] ** 2).sum()

    # Compute mean
    r_mean = r_sum / num_pixels
    g_mean = g_sum / num_pixels
    b_mean = b_sum / num_pixels

    # Compute standard deviation
    r_std = ((r_sq_sum / num_pixels) - r_mean ** 2) ** 0.5
    g_std = ((g_sq_sum / num_pixels) - g_mean ** 2) ** 0.5
    b_std = ((b_sq_sum / num_pixels) - b_mean ** 2) ** 0.5

    print("Mean:", round(r_mean.item(), 4), round(g_mean.item(), 4), round(b_mean.item(), 4))
    print("Std:", round(r_std.item(), 4), round(g_std.item(), 4), round(b_std.item(), 4))

def calculate_mean_std_labels(datadirs, gene, patients):
    for patient in patients:
        bins = 30
        print("patient", patient)
        print(data_dir)
        loader = get_base_dataset(data_dir, [gene], samples = [patient],)
        print("len(loader)", len(loader))
        counts, bin_edges = np.histogram(loader[gene], bins=bins)
        print(counts)
        print(bin_edges)
        plt.title(patient + " " + gene)
        plt.hist(loader[gene], bins=bins)
        plt.show()
        print("std", np.std(loader[gene]))
        print("mean", np.mean(loader[gene]))

patients_test = ["p008", "p021", "p026"]
data_dir_test = "../../data/crc_base/Test_Data/"
def get_patients_datadir(switch, use_val):
    if not switch:
        patients = ["TENX92", "TENX91", "TENX90", "TENX89", "TENX70", "TENX49", "ZEN49", "ZEN48", "ZEN47", "ZEN46",
                    "ZEN45", "ZEN44"]
        if use_val:
            val_patitents = ["TENX29", "ZEN43", "ZEN42", "ZEN40", "ZEN39", "ZEN38", "ZEN36"]
            patients.extend(val_patitents)
        data_dir = "../../data/CRC-N19/"
    else:
        patients = ["p007", "p014", "p016", "p020", "p025"]
        if use_val:
            patients.extend(["p009", "p013"])
        data_dir = "../../data/crc_base/Training_Data/"
    return patients, data_dir
calculate_target_mean_std = True
patients, data_dir = get_patients_datadir(use_base, use_val)
if calculate_target_mean_std:
    for patient in patients:
        bins = 30
        print("patient", patient)
        print(data_dir)
        loader = get_base_dataset(data_dir, [gene], samples = [patient],)
        print("len(loader)", len(loader))
        counts, bin_edges = np.histogram(loader[gene], bins=bins)

        plt.title(patient + " " + gene)
        plt.hist(loader[gene], bins=bins)
        plt.show()
        print("std", np.std(loader[gene]))
        print("mean", np.mean(loader[gene]))


patients, data_dir = get_patients_datadir(use_base, use_val)



exit(0)







patients, data_dir = get_patients_datadir(switch, use_val)

r, g, b = 0,0,0
len_datasets = 0
for patient in patients:
    dataset = get_dataset(data_dir, [gene], samples=[patient], transforms=get_transforms())
    print("patient", patient)
    len_dataset = len(dataset)
    for idx, (data, target) in enumerate(dataset):
        r += data[0].mean()
        g += data[1].mean()
        b += data[2].mean()
    len_datasets += len_dataset


print("mean", r/len_datasets, g/len_datasets, b/len_datasets)

if use_test:
    for patient in patients_test:
        dataset = get_dataset(data_dir_test, [gene], samples=[patient], transforms=get_transforms())
        print("patient", patient)
        len_dataset = len(dataset)
        for idx, (data, target) in enumerate(dataset):
            r += data[0].mean()
            g += data[1].mean()
            b += data[2].mean()
        len_datasets += len_dataset