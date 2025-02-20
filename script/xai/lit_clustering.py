import torch
print(torch.__version__)
from script.xai.cluster_functions import cluster_explanations_genes_loop
import wandb

def cluster(model, data_dir, samples, genes, out_dir, debug=False):
    print("clustering start")
    results_paths = cluster_explanations_genes_loop(model, data_dir, out_dir, genes=genes, debug=debug, samples=samples)
    for i in range(len(results_paths)):
        wandb.log({"clustering restults for " + genes[i]: wandb.Image(results_paths[i])})
    print("clustering done")
