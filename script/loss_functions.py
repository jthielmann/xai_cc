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
