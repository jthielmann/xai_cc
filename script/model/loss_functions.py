
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import pearson_corrcoef
from typing import Dict, Union

class SparsityLoss(nn.Module):
    def __init__(self, layername, model):
        super(SparsityLoss, self).__init__()
        self.activations = None
        for name, module in model.named_modules():
            if name == layername:
                module.register_forward_hook(self.fw_hook)

    def fw_hook(self, module, input, output):
        self.activations = output.clone()

    def forward(self, out, label):
        a = self.activations.clamp(min=0).amax((2, 3)) / (self.activations.shape[2] * self.activations.shape[3])
        sparsity_loss = (a[:, a.sum(0) > 0].mean())
        return sparsity_loss


def dino_loss(student_output, teacher_output, temperature):
    teacher_out = F.softmax(teacher_output / temperature, dim=-1).detach()
    student_out = F.log_softmax(student_output / temperature, dim=-1)
    return -torch.mean(torch.sum(teacher_out * student_out, dim=-1))


class CompositeLoss(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = nn.ModuleList(losses)  # so they’re part of .parameters()

    def forward(self, out, target):  # pass-through for extra args
        weighted_sum = 0.0
        for idx, loss_fn in enumerate(self.losses):
            # Each loss_fn decides what to do with *extra (e.g. sample weights)
            l = loss_fn(out, target)
            weighted_sum += l
        return weighted_sum



class MultiGeneWeightedMSE(nn.Module):
    """Strict weighted MSE for multi-gene regression.

    Expects per-sample weights aligned with predictions/targets and will
    raise on any mismatch. No internal binning or fallback logic.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch pred{tuple(pred.shape)} vs target{tuple(target.shape)}")
        if sample_weights is None:
            raise ValueError("sample_weights must be provided for MultiGeneWeightedMSE")
        if sample_weights.shape != pred.shape:
            raise ValueError(
                f"sample_weights must match pred/target shape. Got weights{tuple(sample_weights.shape)} vs pred{tuple(pred.shape)}"
            )

        per_elem = (pred - target) ** 2  # shape (B, G)
        w = sample_weights
        w_sum = torch.clamp(w.sum(), min=self.eps)
        return (per_elem * w).sum() / w_sum


class PearsonCorrLoss(nn.Module):
    def __init__(self, dim: int = 0, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = preds
        y = target
        x = x - x.mean(dim=self.dim, keepdim=True)
        y = y - y.mean(dim=self.dim, keepdim=True)

        xy = (x * y).sum(dim=self.dim)
        xx = (x * x).sum(dim=self.dim)
        yy = (y * y).sum(dim=self.dim)

        r = xy / (torch.sqrt(xx + self.eps) * torch.sqrt(yy + self.eps))  # shape: (G,) or scalar
        loss = 1 - r
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
