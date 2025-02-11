from data_loader import get_dataset
import os
import pandas as pd
import torch
import anndata as ad
import ntpath
from torch.utils.data import DataLoader



def ann_data_to_csv(data_dir):
    patients = [os.path.basename(f) for f in os.scandir(data_dir) if f.is_dir()]

    # generate training dataframe with all training samples
    for i in patients:
        adata = ad.read_h5ad(data_dir + "/" + i + "/meta_data/tmm_combat_scaled_" + i + ".h5ad")
        st_dataset = adata.to_df()
        st_dataset["tile"] = st_dataset.index
        st_dataset['tile'] = st_dataset['tile'].apply(
            lambda x: ntpath.basename(x[0:-2]))
        st_dataset.set_index('tile')
        filename = data_dir + "/" + i + "/meta_data/gene_data.csv"
        st_dataset.to_csv(filename, index=False)

        
def process_spatial_data(patients, data_dir):
    for patient in patients:
        filename = data_dir + "/" + patient + "/meta_data/spatial_data.csv"
        spatial_matrix = pd.read_csv(
            data_dir + patient + "/meta_data/Raw_Spatial_Matrix_156_" + patient + ".csv")  # adapt to sample
        spatial_matrix['tile'] = spatial_matrix['tile'].apply(lambda x: "{}{}".format(x, ".tiff"))
        spatial_matrix['tile'] = spatial_matrix['tile'].apply(
            lambda x: "{}{}".format(data_dir + patient + "/tiles/", x[57:]))  # adapt to sample
        spatial_matrix['path'] = spatial_matrix['tile']
        spatial_matrix['tile'] = spatial_matrix['tile'].apply(lambda x: os.path.basename(x))
        spatial_matrix.to_csv(filename, index=False)


def generate_results_patient(model, device, data_dir, patient, genes, filepath, transform=None, max_len=None):
    model.eval()
    print("generating results for patient", patient)
    dataset = get_dataset(data_dir, genes, transform, [patient], max_len=max_len)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    columns = []

    for gene in genes:
        columns.append("labels_" + gene)

    for gene in genes:
        columns.append("out_" + gene)

    columns.append("path")
    columns.append("tile")
    columns.append("patient")
    if not os.path.exists(filepath):
        df = pd.DataFrame(columns=columns)
        df.to_csv(filepath, index=False)
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            images = images.float()
            labels = labels.clone().detach().to(device)

            output = model(images)

            output = pd.DataFrame(output.cpu())
            labels = pd.DataFrame(labels.cpu())
            name = loader.dataset.get_tilename(idx)
            res = pd.concat([labels, output, pd.Series(name), pd.Series(os.path.basename(name)), pd.Series(patient)], axis=1)
            res.columns = columns

            res.to_csv(filepath, index=False, mode='a', header=False)
    return filepath, columns


def merge_data(data_dir, patient, results_filename="results.csv", squelch=True):
    base_path = data_dir + patient + "/meta_data/"
    spatial_matrix = pd.read_csv(base_path + "spatial_data.csv")
    results = pd.read_csv(base_path + results_filename)
    results = results.drop("path", axis=1)
    # Merge spatial with test dataset to get spatial coordinates for predicted tiles
    merge = pd.merge(results, spatial_matrix, on="tile")
    merge.set_index("tile", inplace=True)
    merge.to_csv(base_path + "merge.csv")
    if not squelch:
        print(base_path + "merge.csv")
        print(base_path + results_filename)


def merge_gene_data_and_coords(data_dir, patient, results_dir, filename, gene):
    base_path = data_dir + patient + "/meta_data/"
    spatial_matrix = pd.read_csv(base_path + "spatial_data.csv")
    gene_data      = pd.read_csv(base_path + "gene_data.csv")
    gene_columns  = ['tile', gene]
    gene_data = gene_data[gene_columns]
    merge = pd.merge(gene_data, spatial_matrix, on="tile")
    merge.set_index("tile", inplace=True)
    results_path = results_dir + "/" + patient + "/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    merge.to_csv(results_path + filename)


def generate_results2(model, device, data_dir, results_dir,patient=None, gene="RUBCNL", results_filename=None):#
    if results_filename is None:
        results_filename = gene + "_results.csv"
    model.eval()
    loader = get_patient_loader(data_dir, patient, gene)
    filename = results_dir + data_dir + patient + results_filename
    if os.path.exists(filename):
        file_df = pd.read_csv(filename)
    columns = ['labels', 'output', 'path', 'tile']
    df = pd.DataFrame(columns=columns)
    df.to_csv(filename, index=False)
    i = 0
    j = len(loader)
    with torch.no_grad():
        for images, labels, name in loader:
            if i % 1000 == 0:
                print(i, "/", j)
            i += 1
            images = images.unsqueeze(0).to(device)
            images = images.float()
            labels = torch.tensor(labels[0]).to(device)

            output = model(images)

            output = output.squeeze().unsqueeze(0)
            labels = labels.unsqueeze(0)

            output = pd.DataFrame(output.cpu())
            labels = pd.DataFrame(labels.cpu())

            res = pd.concat([labels, output, pd.Series(name), pd.Series(os.path.basename(name))], axis=1)
            res.columns = columns

            res.to_csv(filename, index=False, mode='a', header=False)