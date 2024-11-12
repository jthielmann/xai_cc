image_dims = (224, 224)
import random
def patch_coords(image_size, patchsize):
    x = int(random.random() * (image_size[0] - patchsize))
    y = int(random.random() * (image_size[1] - patchsize))

    return x, y

for i in range(10):
    print(patch_coords(image_dims, 32))

