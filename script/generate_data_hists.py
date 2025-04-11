import sys

sys.path.insert(0, '..')
from script.data_processing.data_loader import get_base_dataset, get_dataset
import matplotlib.pyplot as plt
import numpy as np
import os


gene = "RUBCNL"
dataset_name = "pseudospot"
from script.data_processing.image_transforms import get_transforms
use_val = False
use_test = False
def get_mean_std_features(data_dirs, patients, genes=None, gene_data_filename="gene_data.csv"):
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
            dataset = get_dataset(data_dir, genes, samples=[patient], transforms=get_transforms(),gene_data_filename=gene_data_filename)
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
    else: # CRC-N19_2
        patients = ["TENX92", "TENX91", "TENX90", "TENX89", "TENX70", "TENX49", "ZEN49", "ZEN48", "ZEN47", "ZEN46",
                    "ZEN45", "ZEN44"]
        if use_val:
            val_patitents = ["TENX29", "ZEN43", "ZEN42", "ZEN40", "ZEN39", "ZEN38", "ZEN36"]
            patients.extend(val_patitents)
        data_dir = "../data/N19/"
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
#get_mean_std_features([data_dir], patients, genes=[gene], gene_data_filename=gene_data_filename)



r, g, b = 0,0,0
len_datasets = 0
for patient in patients:
    dataset = get_dataset(data_dir, [gene], samples=[patient], transforms=get_transforms(normalize=False),gene_data_filename=gene_data_filename)
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
        dataset = get_dataset(data_dir_test, [gene], samples=[patient], transforms=get_transforms(),gene_data_filename=gene_data_filename)
        print("patient", patient)
        len_dataset = len(dataset)
        for idx, (data, target) in enumerate(dataset):
            r += data[0].mean()
            g += data[1].mean()
            b += data[2].mean()
        len_datasets += len_dataset