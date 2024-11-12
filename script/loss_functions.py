import torch
import torch.nn as nn

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


class CompositeLoss(nn.Module):
    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses = losses
        # Use Parameter for weights if they need to be optimized; otherwise set requires_grad=True on tensors
        if weights is not None:
            self.weights = nn.ParameterList(
                [nn.Parameter(w) if isinstance(w, torch.Tensor) else nn.Parameter(torch.tensor(w))
                 for w in weights])
        else:
            self.weights = [torch.tensor(1 / len(losses)) for _ in losses]

        # Check that the number of weights matches the number of losses
        if len(self.weights) != len(self.losses):
            raise ValueError("CompositeLoss: Number of weights must match number of losses")

    def forward(self, out, label):
        # Accumulate the weighted loss terms
        loss = sum(self.losses[i](out, label) * self.weights[i] for i in range(len(self.losses)))
        return loss


class CompositeLoss(nn.Module):
    def __init__(self, losses, weights = None):
        super().__init__()
        self.losses = losses
        if weights:
            self.weights = weights
        else:
            self.weights = [torch.tensor(1/len(losses)) for _ in losses]
        if len(self.weights) != len(self.losses):
            print(len(self.weights), len(self.losses), len(losses), len(weights) if weights else "weights is None")
            raise ValueError("CompositeLoss: Number of weights must match number of losses")

    def forward(self, out, label):
        return torch.sum(torch.tensor([self.losses[i](out, label)*self.weights[i] for i in range(len(self.losses))]))



