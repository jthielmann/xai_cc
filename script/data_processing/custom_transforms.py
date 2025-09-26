import torch
import torchvision.transforms.v2 as T


def patch_coords(image_size, patch_width, patch_height)-> (int,int):
    assert patch_width <= image_size[0]
    assert patch_height <= image_size[1]
    x = torch.randint(0, image_size[0] - patch_width + 1, (1,)).item()
    y = torch.randint(0, image_size[1] - patch_height + 1, (1,)).item()
    return x, y


# to be used as a transform in a torchvision.transforms.Compose
class Occlude(T.Transform):
    def __init__(self, patch_size_x, patch_size_y, patch_vary_width=0, patch_min_size=10, patch_max_size=100,
                 use_batch=True):
        super().__init__()
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.patch_vary_width = patch_vary_width
        self.patch_min_size = patch_min_size
        self.patch_max_size = patch_max_size
        self.use_batch = use_batch

    def __call__(self, sample: torch.Tensor)-> torch.Tensor:
        h, w = sample.size()[-2:]
        patch_size_x = self.patch_size_x
        patch_size_y = self.patch_size_y
        if self.patch_vary_width != 0:
            patch_size_x += torch.randint(-self.patch_vary_width, self.patch_vary_width, (1,)).item()
            patch_size_y += torch.randint(-self.patch_vary_width, self.patch_vary_width, (1,)).item()
            patch_size_x = max(self.patch_min_size, patch_size_x)
            patch_size_y = max(self.patch_min_size, patch_size_y)
            patch_size_x = min(self.patch_max_size, patch_size_x)
            patch_size_y = min(self.patch_max_size, patch_size_y)
        x, y = patch_coords((h, w), patch_size_x, patch_size_y)
        if self.use_batch:
            sample[:, :, x:x + patch_size_x, y:y + patch_size_y] = 0
        else:
            sample[:, x:x + patch_size_x, y:y + patch_size_y] = 0

        return sample
