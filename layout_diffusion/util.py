import importlib
import mindspore.communication.management as dist  # mindspore distributed module
import numpy as np
import mindspore as ms  # mindspore module
import os
import random


def get_obj_from_str(string, reload=False):
    # This function returns an object from a string
    module, cls = string.rsplit(".", 1)  # split the string by the last dot
    if reload:
        module_imp = importlib.import_module(module)  # import the module
        importlib.reload(module_imp)  # reload the module

    return getattr(importlib.import_module(module, package=None), cls)  # return the object


def fix_seed(seed=2333):
    # This function fixes the random seed for reproducibility
    if dist.get_rank() > 0:  # if distributed mode is on
        seed = seed + dist.get_rank()  # add the rank to the seed

    np.random.seed(seed)  # set numpy random seed
    ms.set_seed(seed)  # set mindspore random seed for CPU and GPU

    random.seed(seed)  # set python random seed
    np.random.seed(seed)  # set numpy random seed again

    os.environ['PYTHONHASHSEED'] = str(seed)  # set python hash seed


def loopy(data_loader):
    # This function loops over the data loader infinitely
    while True:
        for x in iter(data_loader):  # iterate over data loader
            yield x  # yield data batch
