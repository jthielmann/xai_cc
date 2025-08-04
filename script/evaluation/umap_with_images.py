import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import json
import sys
sys.path.insert(0, '..')
from script.model.lit_model import load_model
from script.data_processing.data_loader import get_dataset_for_umap
from script.data_processing.image_transforms import get_transforms
import torch.nn.functional as F

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform, image_size=224):
        self.image_paths = image_paths
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image and convert to RGB
        img = Image.open(self.image_paths[idx]).convert("RGB")
        # Resize using PIL and then convert to numpy array (for plotting)
        original = np.array(img.resize((self.image_size, self.image_size)))
        if self.transform:
            img = self.transform(img)
        return img, original


def create_umap_embedding_efficient(model, loader, batch_size=32, device='cuda', components=2, n_neighbors=15):
    model = model.to(device)
    model.eval()

    features_list = []
    paths_list = []  # For plotting: stores original images
    with torch.no_grad():
        for imgs, paths in loader:
            imgs = imgs.to(device)
            # Extract features (assumes model returns feature vectors)
            features = model(imgs)
            features_list.append(features.cpu().numpy())
            paths_list.extend(paths)

    # Concatenate features from all batches
    features_np = np.concatenate(features_list, axis=0)

    # Compute the UMAP embedding on the aggregated features
    umap_model = UMAP(n_components=components, random_state=42, n_neighbors=n_neighbors)
    embedding = umap_model.fit_transform(features_np)
    return embedding, paths_list


def plot_umap_with_images(embedding, tile_paths, title, zoom=0.1, output_path="../out/", components=2):
    fig, ax = plt.subplots(figsize=(10, 10))

    if components == 2:
        for (x, y), path in zip(embedding, tile_paths):
            # Open the image and convert to RGB
            img = Image.open(path).convert("RGB")
            # Apply your transform pipeline; assume it returns a tensor in shape (C, H, W)
            img_tensor = get_transforms(normalize=False)(img)

            # Get current dimensions from the tensor shape (C, H, W)
            C, H, W = img_tensor.shape
            # Compute new dimensions based on the zoom factor
            new_H = int(H * zoom)
            new_W = int(W * zoom)
            # Resize the tensor using interpolation; note that interpolate expects a batched input
            img_tensor_resized = F.interpolate(img_tensor.unsqueeze(0), size=(new_H, new_W), mode='bilinear',
                                               align_corners=False)
            img_tensor_resized = img_tensor_resized.squeeze(0)

            # Convert the resized tensor to a NumPy array and adjust dimensions for OffsetImage:
            # PyTorch tensors are (C, H, W), but OffsetImage expects (H, W, C)
            img_np = img_tensor_resized.cpu().numpy().transpose(1, 2, 0)

            imagebox = OffsetImage(img_np, zoom=zoom)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)
    elif components == 3:
        from mpl_toolkits.mplot3d import proj3d

        for (x, y, z), path in zip(embedding, tile_paths):
            img = Image.open(path)
            img = get_transforms()(img)
            # Optionally, convert to a NumPy array if needed:
            img = np.array(img)

            # Project the 3D point to 2D
            x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
            imagebox = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(imagebox, (x2, y2), frameon=False)
            ax.add_artist(ab)
    else:
        raise ValueError("components must be 2 or 3")
    ax.set_xlim(embedding[:, 0].min() - 1, embedding[:, 0].max() + 1)
    ax.set_ylim(embedding[:, 1].min() - 1, embedding[:, 1].max() + 1)

    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(output_path + title + ".png")
    plt.show()

def custom_collate_fn(batch):
    # batch is a list of tuples: (transformed_img, original)
    imgs = torch.stack([item[0] for item in batch])
    originals = [item[1] for item in batch]  # Keep as a list
    return imgs, originals


# Example usage:
if __name__ == "__main__":
    import os
    from torchvision import models

    # Load a encoder ResNet50 and modify it to output features.
    resnet50 = models.resnet50(encoder=True)
    resnet50.fc = torch.nn.Identity()  # Remove final classification layer
    model_path = "../models/bins 10/ResNet_ep_40_lr_0.01_resnet50random_MSELoss_False_CRC_N19_32_RUBCNL/best_model.pth"
    config_path = "../models/bins 10/ResNet_ep_40_lr_0.01_resnet50random_MSELoss_False_CRC_N19_32_RUBCNL/config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    model = load_model(model_path, config)
    # Directory containing your images
    path_to_images = "../data/crc_base/Training_Data/p007/tiles/"  # update with your directory path
    image_paths = [os.path.join(path_to_images, fname)
                   for fname in os.listdir(path_to_images)
                   if fname.lower().endswith(('.png', '.jpg', '.jpeg', 'tiff', 'tif'))]

    samples = ["p007", "p014", "p016", "p020", "p025"]
    dataset = get_dataset_for_umap("../data/crc_base/Training_Data/", ["RUBCNL"], transforms=get_transforms(), samples=samples)
    loader = DataLoader(dataset, batch_size=32, num_workers=0, collate_fn=custom_collate_fn, shuffle=False)
    # Compute the UMAP embedding using the memory-efficient function
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    for i in range (2):
        for ii in range(20):
            components = i + 3
            neighbors = ii + 2

            embedding, tile_paths = create_umap_embedding_efficient(model, loader, batch_size=32, device=device, components=components, n_neighbors=neighbors)

            # Plot the UMAP embedding with images as markers
            plot_umap_with_images(embedding, tile_paths, title="components" + str(components) + "neighbors" + str(neighbors), zoom=0.3, components=components)
