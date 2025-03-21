import matplotlib.pyplot as plt
import pandas as pd
import scipy
from PIL import Image


def plot_data_mapped(data_dir, patient, category="diff"):
    base_path = data_dir + patient + "/meta_data/"
    merge = pd.read_csv(base_path + "merge.csv")
    merge["diff"] = merge["labels"] - merge["output"]
    merge.to_csv(base_path + "merge.csv")
    plt.scatter(merge['x'], merge['y'], c=merge[category], cmap='viridis')
    plt.colorbar(label='labels')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(category + " for patient " + patient)
    plt.show()


def plot_data_scatter(data_dir, patient, genes):
    base_path = data_dir+patient+"/meta_data/"
    merge = pd.read_csv(base_path + "merge.csv")
    plt.scatter(merge['output'], merge['labels'])
    plt.text(x=-2, y=3 , s="pearson corr: " + str(round(scipy.stats.pearsonr(merge['output'], merge['labels'])[0], 2)))
    plt.plot( [-2,3],[-2,3], color='red' )
    title = "labels vs output for " + patient + " for genes"
    for gene in genes:
        title += gene 
    plt.title(title)
    plt.xlabel('output')
    plt.ylabel('target')
    plt.show()


def plot_hist_comparison(data_dir, patient):
    base_path = data_dir + patient + "/meta_data/"
    merge = pd.read_csv(base_path + "merge.csv")
    out = merge['output']
    plt.hist(out)
    plt.show()
    labels = merge['labels']
    plt.hist(labels)
    plt.show()


def print_metrics(data_dir, patient, metric):
    base_path = data_dir + patient + "/meta_data/"
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
    if metric == "corr":
        print("corr: ", scipy.stats.pearsonr(out, labels)[0])


def plot_tile(tile_path):
    img = Image.open(tile_path)
    plt.imshow(img)
    plt.show()
