import matplotlib as plt
import pandas as pd
import scipy


def plot_data_mapped(data_dir, patient, category="diff"):
    base_path = data_dir + patient + "/Preprocessed_STDataset/"
    merge = pd.read_csv(base_path + "merge.csv")
    merge["diff"] = merge["labels"] - merge["output"]
    merge.to_csv(base_path + "merge.csv")
    plt.scatter(merge['x'], merge['y'], c=merge[category], cmap='viridis')
    plt.colorbar(label='labels')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(category + " for patient " + patient)
    plt.show()
    # merge.plot.scatter(x="x", y="y")


def plot_hist_comparison(data_dir, patient):
    base_path = data_dir + patient + "/Preprocessed_STDataset/"
    merge = pd.read_csv(base_path + "merge.csv")
    out = merge['output']
    plt.hist(out)
    plt.show()
    labels = merge['labels']
    plt.hist(labels)
    plt.show()


def print_metrics(data_dir, patient, metric):
    base_path = data_dir + patient + "/Preprocessed_STDataset/"
    merge = pd.read_csv(base_path + "merge.csv")
    out = merge['output']
    labels = merge['labels']
    mean_out = out.mean()
    std_out = out.std()
    mean_labels = labels.mean()
    std_labels = labels.std()
    if metric == "mean":
        print("mean diff: ", "{:.4f}".format(abs(mean_out - mean_labels)), ", mean out: ", "{:.4f}".format(mean_out),
              ", mean labels: ", "{:.4f}".format(mean_labels))
    if metric == "std":
        print("std diff: ", "{:.4f}".format(abs(std_out - std_labels)), ", std out: ", "{:.4f}".format(std_out),
              ", std labels: ", "{:.4f}".format(std_labels))


def plot_hist_comparison(data_dir, patient):
    base_path = data_dir + patient + "/Preprocessed_STDataset/"
    merge = pd.read_csv(base_path + "merge.csv")
    out = merge['output']
    plt.hist(out)
    plt.show()
    labels = merge['labels']
    plt.hist(labels)
    plt.show()


def print_metrics(data_dir, patient, metric):
    base_path = data_dir + patient + "/Preprocessed_STDataset/"
    merge = pd.read_csv(base_path + "merge.csv")
    out = merge['output']
    labels = merge['labels']
    mean_out = out.mean()
    std_out = out.std()
    mean_labels = labels.mean()
    std_labels = labels.std()
    if metric == "mean":
        print("mean diff: ", "{:.4f}".format(abs(mean_out - mean_labels)), ", mean out: ", "{:.4f}".format(mean_out),
              ", mean labels: ", "{:.4f}".format(mean_labels))
    if metric == "std":
        print("std diff: ", "{:.4f}".format(abs(std_out - std_labels)), ", std out: ", "{:.4f}".format(std_out),
              ", std labels: ", "{:.4f}".format(std_labels))
    if metric == "pearson":
        print("pearson corr: " + str(round(scipy.stats.pearsonr(out, labels)[0], 2)))