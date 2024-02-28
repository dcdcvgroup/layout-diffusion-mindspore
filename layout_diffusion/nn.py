import math

import mindspore
import mindspore as ms
from mindspore import nn, ops, jit, Parameter


class GroupNorm32(nn.GroupNorm):
    def construct(self, x: ms.Tensor):
        dims = x.ndim
        if dims == 4:
            return super().construct(x.float()).astype(x.dtype)
        elif dims == 3:
            return super().construct(x.unsqueeze(-1).float()).astype(x.dtype).squeeze(-1)
        raise ValueError(f"unsupported dimensions: {dims}")


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Dense(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


# Instead, use nn.conv_nd(in_channel, out_channel, kernel_size, weight_init='Zero')
# def zero_module(module):
#     """
#     Zero out the parameters of a module and return it.
#     """
#     for p in module.parameters():
#         p.set_data(ops.functional.zeros_like(p.data))
#     return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(axis=list(range(1, len(tensor.shape))))


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2

    freqs = ops.exp(
        -math.log(max_period) * ops.arange(
            start=0, end=half, dtype=ms.float32
        ) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = ops.cat([ops.Cos()(args), ops.Sin()(args)], axis=-1)
    if dim % 2:
        embedding = ops.cat([embedding, ops.ZerosLike()(embedding[:, :1])], axis=-1)
    return embedding
