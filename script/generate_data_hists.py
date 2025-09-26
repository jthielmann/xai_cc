import sys

import torch

sys.path.insert(0, '..')
from script.data_processing.data_loader import get_base_dataset, get_dataset, NCT_CRC_Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import transforms

gene = "RUBCNL"
dataset_name = "NCT-CRC-HE-100K"
from script.data_processing.transforms import get_encoder_transforms
use_val = False
use_test = False

def get_mean_std_per_patient(
    dataset_name,
    data_dirs,
    gene,
    patients,
    use_tiles_subdir=False,
    gene_data_filename=None,
    batch_size=64,
    num_workers=4,
):
    to_tensor = transforms.ToTensor()
    results = {}  # patient -> (mean[C], std[C])

    for patient in patients:
        # build ONE dataset per data_dir (filtered to this patient), then concat
        per_dir_datasets = []
        for data_dir in data_dirs:
            # quick existence check (optional; dataset classes may already filter)
            img_dir = os.path.join(data_dir, patient, "tiles" if use_tiles_subdir else "")
            if not os.path.exists(img_dir):
                continue

            if dataset_name == "NCT-CRC-HE-100K":
                ds = NCT_CRC_Dataset(
                    data_dir,
                    [patient],
                    use_tiles_sub_dir=use_tiles_subdir,
                    image_transforms=to_tensor,
                )
            else:
                ds = get_dataset(
                    data_dir,
                    [gene],
                    samples=[patient],
                    transforms=to_tensor,  # no augmentation because we analyze data
                    gene_data_filename=gene_data_filename,
                )
            if len(ds) > 0:
                per_dir_datasets.append(ds)

        if not per_dir_datasets:
            print(f"[warn] patient {patient} not found in any data_dir")
            continue

        dataset = per_dir_datasets[0] if len(per_dir_datasets) == 1 else torch.utils.data.ConcatDataset(per_dir_datasets)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

        sum_c   = torch.zeros(3, dtype=torch.float64)
        sumsq_c = torch.zeros(3, dtype=torch.float64)
        total_pixels = 0

        with torch.no_grad():
            for batch in loader:
                data = batch[0]  # take data from (data, label), [B, 3, H, W]
                if data is None or data.ndim != 4 or data.size(1) != 3:
                    continue
                b, _, h, w = data.shape
                total_pixels += b * h * w

                d64 = data.to(torch.float64)  # reduce rounding issues
                sum_c   += d64.sum(dim=(0, 2, 3))
                sumsq_c += (d64 ** 2).sum(dim=(0, 2, 3))

        if total_pixels == 0:
            print(f"[warn] patient {patient} has zero pixels after filtering")
            continue

        mean = (sum_c / total_pixels).to(torch.float32)
        var  = (sumsq_c / total_pixels) - (mean.to(torch.float64) ** 2)
        std  = var.clamp_min(0).sqrt().to(torch.float32)

        print(f"Patient {patient} Mean: {[round(x.item(),4) for x in mean]}, Std: {[round(x.item(),4) for x in std]}")
        results[patient] = (mean, std)

    return results

def calculate_mean_std_labels(data_dir, gene, patients, filename="gene_data.csv"):
    bins = 30
    print("patient", patients)
    print(data_dir)
    loader = get_base_dataset(data_dir, [gene], samples = patients,gene_data_filename=filename)
    print("len(loader)", len(loader))
    counts, bin_edges = np.histogram(loader[gene], bins=bins)
    print(counts)
    print(bin_edges)
    title = ""
    for i in patients:
        title += i + " "
    title += gene
    plt.title(title)
    plt.hist(loader[gene], bins=bins)
    plt.show()
    print("std", np.std(loader[gene]))
    print("mean", np.mean(loader[gene]))




patients_test = ["p008", "p021", "p026"]
data_dir_test = "../data/crc_base/Test_Data/"
def get_patients_datadir(dataset_name, use_val):
    if dataset_name == "CRC-N19":
        patients = ["TENX92", "TENX91", "TENX90", "TENX89", "TENX70", "TENX49", "ZEN49", "ZEN48", "ZEN47", "ZEN46",
                    "ZEN45", "ZEN44"]
        if use_val:
            val_patitents = ["TENX29", "ZEN43", "ZEN42", "ZEN40", "ZEN39", "ZEN38", "ZEN36"]
            patients.extend(val_patitents)
        data_dir = "../data/CRC-N19/"
    elif dataset_name == "crc_base":
        patients = ["p007", "p014", "p016", "p020", "p025"]
        if use_val:
            patients.extend(["p009", "p013"])
        data_dir = "../data/crc_base/Training_Data/"
    elif dataset_name == "pseudospot":
        patients = ["p007", "p014", "p016", "p020", "p025"]
        if use_val:
            patients.extend(["p009", "p013"])
        data_dir = "../data/pseudospot/"
    elif dataset_name == "CRC-N19_2":
        patients = ["TENX92", "TENX91", "TENX90", "TENX89", "TENX70", "TENX49", "ZEN49", "ZEN48", "ZEN47", "ZEN46",
                    "ZEN45", "ZEN44"]
        if use_val:
            val_patitents = ["TENX29", "ZEN43", "ZEN42", "ZEN40", "ZEN39", "ZEN38", "ZEN36"]
            patients.extend(val_patitents)
        data_dir = "../data/N19/"
    elif dataset_name == "NCT-CRC-HE-100K":
        patients = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
        data_dir = "../data/NCT-CRC-HE-100K/"
    else:
        print("dataset not recognized")
        exit(1)

    return patients, data_dir

calculate_target_mean_std = False
patients, data_dir = get_patients_datadir(dataset_name, use_val)
gene_data_filename = "gene_data_ranknorm.csv"
#print(calculate_mean_std_labels(data_dir, gene, patients,gene_data_filename))
if calculate_target_mean_std:
    for patient in patients:
        bins = 10
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
calculate_hist_whole_dataset = False
if calculate_hist_whole_dataset:
    bins = 30
    print("patients", patients)
    print(data_dir)
    loader = get_base_dataset(data_dir, [gene], samples=patients, )
    print("len(loader)", len(loader))
    counts, bin_edges = np.histogram(loader[gene], bins=bins)

    #plt.title("patients" + " " + gene)
    plt.hist(loader[gene], bins=bins)
    #plt.show()
    print("std", np.std(loader[gene]))
    print("mean", np.mean(loader[gene]))

    def get_bin_idx(label, bin_edges):
        # define your own bin_index function
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= label < bin_edges[i + 1]:
                bin_idx = i
                return bin_idx
        return len(bin_edges) - 1
    from collections import Counter
    from scipy.ndimage import convolve1d
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal.windows import triang

    def get_lds_kernel_window(kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(
                gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks)
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
                map(laplace, np.arange(-half_ks, half_ks + 1)))

        return kernel_window


    def get_lds_kernel_window_silverman(ks):
        """
        Generate a kernel window using a Gaussian KDE with Silverman's rule.

        Parameters:
            ks (int): The kernel window size (should be odd for symmetry).

        Returns:
            np.ndarray: A 1D numpy array of length ks containing the normalized kernel window.
        """
        # Ensure ks is an odd number to have a symmetric window
        if ks % 2 == 0:
            raise ValueError("Kernel size ks must be odd for a symmetric window.")

        half_ks = (ks - 1) // 2
        # Create an array of positions centered at zero
        positions = np.arange(-half_ks, half_ks + 1, dtype=float)

        # Create a Gaussian KDE over the positions with Silverman's rule for bandwidth
        from scipy.stats import gaussian_kde
        # gaussian_kde expects data in shape (d, n), so reshape positions to (1, ks)
        kde = gaussian_kde(positions[None, :], bw_method='silverman')
        # Evaluate the KDE at the given positions
        kernel_window = kde.evaluate(positions[None, :])

        # Normalize the kernel window so that its maximum value is 1
        kernel_window = kernel_window / np.max(kernel_window)

        return kernel_window

    # preds, labels: [Ns,], "Ns" is the number of total samples
    # assign each label to its corresponding bin (start from 0)
    # with your defined get_bin_idx(), return bin_index_per_label: [Ns,]
    bin_index_per_label = [get_bin_idx(label, bin_edges) for label in loader[gene].values]

    # calculate empirical (original) label distribution: [Nb,]
    # "Nb" is the number of bins
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = counts#[num_samples_of_bins.get(i, 0) for i in range(Nb)]
    ks = 3
    sigma = None
    # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
    #lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=ks, sigma=sigma)
    lds_kernel_window = get_lds_kernel_window_silverman(ks)
    # calculate effective label distribution: [Nb,]
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
    plt.title("ks: " + str(ks) + " sigma: " + str(sigma))
    plt.stairs(eff_label_dist, edges=bin_edges)
    plt.show()
    """print("[")
    for i in eff_label_dist:
        print(str(i) + ",")
    print("]")"""
    eff_str = "["
    for i in eff_label_dist:
        eff_str += str(i) + ", "
    eff_str += "]"
    print(eff_str)



patients, data_dir = get_patients_datadir(dataset_name, use_val)
print("--------------------------")


get_mean_std_features(dataset_name, [data_dir], gene, patients, use_tiles_subdir=True, gene_data_filename=gene_data_filename)

exit(0)

r, g, b = 0,0,0
len_datasets = 0
for patient in patients:
    if dataset_name == "NCT-CRC-HE-100K":
        dataset = NCT_CRC_Dataset(data_dir, patients, use_tiles_sub_dir=True, image_transforms=transforms.Compose([transforms.ToTensor()]))
    else:
        dataset = get_dataset(
            data_dir,
            [gene],
            samples=[patient],
            transforms=get_encoder_transforms("resnet50random"),
            gene_data_filename=gene_data_filename,
        )
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
        dataset = get_dataset(
            data_dir_test,
            [gene],
            samples=[patient],
            transforms=get_encoder_transforms("resnet50random"),
            gene_data_filename=gene_data_filename,
        )
        print("patient", patient)
        len_dataset = len(dataset)
        for idx, (data, target) in enumerate(dataset):
            r += data[0].mean()
            g += data[1].mean()
            b += data[2].mean()
        len_datasets += len_dataset
