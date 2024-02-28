import PIL
from PIL import Image

import numpy as np
import mindspore
import mindspore as ms
from mindspore import ops
from mindspore.dataset import transforms as T
from mindspore.dataset import vision as V
from typing import List, Sequence

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def rescale(x):
    lo, hi = x.min(), x.max()
    return x.sub(lo).div(hi - lo)


def blank(x):
    return x


def image_normalize():
    # return blank
    return V.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def image_unnormalize(rescale_image=False):
    transforms = [
        V.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
        V.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
    ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)


def image_unnormalize_for_mindspore(image):
    """
    Input:
    - imgs: FloatTensor of shape (N, C, H, W) or (C, H, W) giving preprocessed images

    Output:
    - imgs_de: ByteTensor of shape (N, C, H, W) or (C, H, W) giving deprocessed images
      in the range [0, 255]
    """
    unnormalized_image = ops.zeros_like(image)
    i = 0
    for t, m, s in zip(image, IMAGENET_MEAN, IMAGENET_STD):
        unnormalized_image[i] = t.mul(s).add(m)
        i += 1
    return unnormalized_image


def image_unnormalize_batch(imgs, rescale=False):
    """
    Input:
    - imgs: FloatTensor of shape (N, C, H, W) or (C, H, W) giving preprocessed images

    Output:
    - imgs_de: ByteTensor of shape (N, C, H, W) or (C, H, W) giving deprocessed images
      in the range [0, 255]
    """
    imgs_de = []
    if imgs.dim() == 4:
        for i in range(imgs.size(0)):
            img_de = image_unnormalize_for_mindspore(imgs[i])
            # img_de = img_de.mul(255).clamp(0, 255).byte()
            # img_de = img_de.mul(255).clamp(0, 255)
            imgs_de.append(img_de)
        imgs_de = ops.cat(imgs_de, axis=0)
        return imgs_de
    elif imgs.dim() == 3:
        img_de = image_unnormalize_for_mindspore(imgs)
        return img_de
    else:
        raise NotImplementedError
