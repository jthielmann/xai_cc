import random
import torch

def patch_coords(image_size, patch_size_x, patch_size_y):
    x = int(random.randint(0, image_size[0] - patch_size_x))
    y = int(random.randint(0, image_size[1] - patch_size_y))
    return x, y


# to be used as a transform in a torchvision.transforms.Compose
class Occlude(object):
    def __init__(self, patch_size_x, patch_size_y, patch_vary_width=0, patch_min_width=10):
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.patch_vary_width = patch_vary_width
        self.patch_min_width = patch_min_width

    def __call__(self, sample: torch.tensor):
        h, w = sample.size()[-2:]
        patch_size_x = self.patch_size_x
        patch_size_y = self.patch_size_y
        if self.patch_vary_width != 0:
            patch_size_x += random.randint(-self.patch_vary_width, self.patch_vary_width)
            patch_size_y += random.randint(-self.patch_vary_width, self.patch_vary_width)
            patch_size_x = min(self.patch_min_width, patch_size_x)
            patch_size_y = min(self.patch_min_width, patch_size_y)
        x, y = patch_coords((h, w), patch_size_x, patch_size_y)
        sample[:, x:x + patch_size_x, y:y + patch_size_y] = 0

        return sample
