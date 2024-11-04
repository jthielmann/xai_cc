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

    def forward(self):
        a = self.activations.clamp(min=0).amax((2, 3)) / (self.activations.shape[2] * self.activations.shape[3])
        sparsity_loss = (a[:, a.sum(0) > 0].mean())
        return sparsity_loss


class CompositeLoss(nn.Module):
    def __init__(self, losses, weights = None):
        super().__init__()
        self.losses = losses
        if weights:
            self.weights = weights
        else:
            self.weights = [1/len(losses) for _ in losses]
        if len(self.weights) != len(self.losses):
            print(len(self.weights), len(self.losses), len(losses), len(weights) if weights else "weights is None")
            raise ValueError("CompositeLoss: Number of weights must match number of losses")
    def forward(self, out, label):
        losses = []
        for i in range(len(self.losses)):
            losses.append(self.losses[i](out, label)*self.weights[i])
        return sum(losses)
