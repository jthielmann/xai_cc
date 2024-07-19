from data_loader import get_patient_loader
import os
import pandas as pd
import torch
import anndata as ad
import ntpath


def ann_data_to_csv(data_dir):
    patients = [os.path.basename(f) for f in os.scandir(data_dir) if f.is_dir()]

    # generate training dataframe with all training samples
    for i in patients:
        print(i)
        adata = ad.read_h5ad(data_dir + "/" + i + "/Preprocessed_STDataset/tmm_combat_scaled_" + i + ".h5ad")
        st_dataset = adata.to_df()
        st_dataset["tile"] = st_dataset.index
        file_names = st_dataset.tile.iloc[0][0:-2]
        base_name = ntpath.basename(file_names)
        st_dataset['tile'] = st_dataset['tile'].apply(
            lambda x: ntpath.basename(x[0:-2]))
        print(base_name)
        st_dataset.set_index('tile')
        filename = data_dir + "/" + i + "/Preprocessed_STDataset/gene_data.csv"
        print(filename)
        st_dataset.to_csv(filename, index=False)



def generate_results(model, device, criterion, data_dir, patient=None, gene="RUBCNL", results_filename="results.csv"):
    model.eval()
    print("generating results...")
    loader = get_patient_loader(data_dir, patient=patient, gene=gene)
    filename = data_dir + patient + "/Preprocessed_STDataset/" + results_filename
    print(filename)
    if os.path.exists(filename):
        os.remove(filename)
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
            # labels = torch.stack(labels, dim=1)
            # labels = labels.float()
            labels = torch.tensor(labels[0]).to(device)

            output = model(images)

            output = output.squeeze().unsqueeze(0)
            labels = labels.unsqueeze(0)

            loss = criterion(output, labels).cpu()

            output = pd.DataFrame(output.cpu())
            labels = pd.DataFrame(labels.cpu())

            res = pd.concat([labels, output, pd.Series(name), pd.Series(os.path.basename(name))], axis=1)
            res.columns = columns

            res.to_csv(filename, index=False, mode='a', header=False)


def merge_data(data_dir, patient, results_filename="results.csv"):
    if patient == 'p008':
        return
    base_path = data_dir + patient + "/Preprocessed_STDataset/"
    spatial_matrix = pd.read_csv(base_path + "spatial_data.csv")
    results = pd.read_csv(base_path + results_filename)
    results = results.drop("path", axis=1)
    # Merge spatial with test dataset to get spatial coordinates for predicted tiles
    merge = pd.merge(results, spatial_matrix, on="tile")
    merge.set_index("tile", inplace=True)
    merge.to_csv(base_path + "merge.csv")