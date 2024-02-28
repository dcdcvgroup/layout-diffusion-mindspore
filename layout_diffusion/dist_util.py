"""
Helpers for distributed training.
"""

import io
import os
import socket

import mindspore as ms
from mindspore import ops


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        ops.Broadcast(0)((p,))
