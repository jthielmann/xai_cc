import argparse
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Repo path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.model.lit_dino import DINO
from script.data_processing.data_loader import NCT_CRC_Dataset
from torchvision import transforms as T


def get_eval_transform(image_size=224, mean=(0.7406, 0.5331, 0.7059), std=(0.1651, 0.2174, 0.1574)):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def extract_features(model: DINO, loader: DataLoader, pool: str = 'cls', device='cuda' if torch.cuda.is_available() else 'cpu'):
    feats, labels = [], []
    model.eval().to(device)
    with torch.no_grad():
        for imgs, ys in loader:
            imgs = imgs.to(device, non_blocking=True)
            f = model.encode_features(imgs, pool=pool)
            feats.append(f.cpu())
            if torch.is_tensor(ys):
                labels.append(ys.cpu())
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0) if labels else None
    return feats, labels


def train_linear_probe(train_feats, train_labels, val_feats, val_labels, num_classes: int, epochs: int = 10, lr: float = 1e-2):
    clf = nn.Linear(train_feats.size(1), num_classes)
    opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    for ep in range(epochs):
        clf.train()
        logits = clf(train_feats)
        loss = F.cross_entropy(logits, train_labels)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            clf.eval()
            acc = (clf(val_feats).argmax(dim=1) == val_labels).float().mean().item()
        print(f"[linear-probe] epoch {ep+1}/{epochs} loss={loss.item():.4f} val_acc={acc*100:.2f}%")
    return clf


def knn_retrieval(train_feats, train_labels, val_feats, val_labels, k: int = 1):
    # cosine similarity
    train_n = F.normalize(train_feats, dim=1)
    val_n = F.normalize(val_feats, dim=1)
    sims = val_n @ train_n.t()  # (Nv, Nt)
    topk = sims.topk(k, dim=1).indices  # (Nv, k)
    # majority vote among top-k (here k=1 → nearest neighbor)
    preds = torch.mode(train_labels[topk], dim=1).values
    acc = (preds == val_labels).float().mean().item()
    print(f"[knn] top-{k} acc={acc*100:.2f}%")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=str, required=True, help='NCT-CRC root containing class subfolders')
    ap.add_argument('--classes', type=str, nargs='+', required=True, help='Class folder names to include')
    ap.add_argument('--ckpt', type=str, default=None, help='Path to DINO checkpoint (.pth)')
    ap.add_argument('--model-id', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--pool', type=str, default='cls', choices=['cls','mean'])
    ap.add_argument('--dummy-backbone', action='store_true', help='Use dummy backbone (debug only)')
    args = ap.parse_args()

    tfm = get_eval_transform(args.image_size)
    ds_train = NCT_CRC_Dataset(args.data_dir, classes=args.classes, use_tiles_sub_dir=False, image_transforms=tfm, label_as_string=False)
    ds_val = NCT_CRC_Dataset(args.data_dir, classes=args.classes, use_tiles_sub_dir=False, image_transforms=tfm, label_as_string=False)
    # Simple split
    n = len(ds_train)
    idx = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    from torch.utils.data import Subset
    dl_train = DataLoader(Subset(ds_train, train_idx.tolist()), batch_size=args.batch_size, shuffle=False)
    dl_val = DataLoader(Subset(ds_val, val_idx.tolist()), batch_size=args.batch_size, shuffle=False)

    cfg = {
        'epochs': 1,
        'model_id': args.model_id,
        'use_hf_normalize': False,
        'use_dummy_backbone': args.dummy_backbone,
        'debug': True,
    }
    model = DINO(cfg)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(sd, strict=False)

    train_feats, train_labels = extract_features(model, dl_train, pool=args.pool)
    val_feats, val_labels = extract_features(model, dl_val, pool=args.pool)

    print(f"features: train={tuple(train_feats.shape)} val={tuple(val_feats.shape)}")
    _ = train_linear_probe(train_feats, train_labels, val_feats, val_labels, num_classes=len(args.classes), epochs=args.epochs)
    _ = knn_retrieval(train_feats, train_labels, val_feats, val_labels, k=1)


if __name__ == '__main__':
    main()

