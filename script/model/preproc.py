import torch
import torch.nn as nn


class LinearPCATransform(nn.Module):
    def __init__(self, mean: torch.Tensor, components: torch.Tensor, var: torch.Tensor, whiten: bool, eps: float = 1e-8):
        super().__init__()
        if mean.ndim != 1:
            raise ValueError(f"mean must be 1D, got {tuple(mean.shape)}")
        if components.ndim != 2:
            raise ValueError(f"components must be 2D, got {tuple(components.shape)}")
        if var.ndim != 1:
            raise ValueError(f"var must be 1D, got {tuple(var.shape)}")
        if components.size(1) != mean.size(0):
            raise ValueError("components second dim must equal mean dim")
        if components.size(0) != var.size(0):
            raise ValueError("components rows must equal var size")
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("proj", components.t().detach().clone())  # [d_in, k]
        self.register_buffer("scale", (var + eps).rsqrt().detach().clone())  # [k]
        self.whiten = bool(whiten)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            x = x.view(x.size(0), -1)
        if x.size(1) != self.mean.size(0):
            raise ValueError(f"x dim {x.size(1)} != mean dim {self.mean.size(0)}")
        x_centered = x - self.mean
        y = x_centered @ self.proj  # [B, k]
        if self.whiten:
            y = y * self.scale
        return y

