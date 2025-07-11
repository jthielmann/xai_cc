import torch
import torch.nn as nn
import torch.nn.functional as F

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


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Sequence


class MultiGeneWeightedMSE(nn.Module):
    def __init__(self, gene_weight_arr: Dict[str, Union[float, torch.Tensor]]):
        super().__init__()
        self.genes = list(gene_weight_arr.keys())

        # register every weight tensor so it moves with .to(device)
        for gene, w in gene_weight_arr.items():
            w_tensor = torch.as_tensor(w, dtype=torch.float32)
            self.register_buffer(f"{gene}_weights", w_tensor)

    def _get_pred_tgt(self, x, gene, idx):
        """Return (tensor for this gene) for the two possible input layouts."""
        if isinstance(x, dict):
            return x[gene].view(-1)  # (B,)
        else:  # assume matrix layout (B, G)
            return x[:, idx].view(-1)

    def _sample_weights(self, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Map each target value in `t` to its bin weight from `w`.
        Assumes:
          • w has length K  (one weight per bin)
          • bins are EQUALLY spaced between min(t) and max(t)   (≈ the cache)
        """
        K        = w.numel()
        edges    = torch.linspace(t.min(), t.max(), K + 1, device=t.device)
        bin_idx  = torch.bucketize(t, edges, right=True) - 1
        bin_idx  = bin_idx.clamp_(0, K - 1)
        return w[bin_idx]                      # (B,)

    def forward(self, pred, target):
        losses = []
        for idx, gene in enumerate(self.genes):
            p = self._get_pred_tgt(pred,   gene, idx)    # (B,)
            t = self._get_pred_tgt(target, gene, idx)    # (B,)
            w_vec = getattr(self, f"{gene}_weights")     # (K,) or scalar

            if w_vec.ndim == 0:                          # scalar
                w_sample = w_vec
            elif w_vec.numel() == t.numel():             # already per-sample
                w_sample = w_vec.view_as(t)
            else:                                        # (K,) – need lookup
                w_sample = self._sample_weights(t, w_vec)

            per_elem = F.mse_loss(p, t, reduction="none")
            losses.append((per_elem * w_sample).mean())

        return losses[0] if len(losses) == 1 else torch.stack(losses).mean()



