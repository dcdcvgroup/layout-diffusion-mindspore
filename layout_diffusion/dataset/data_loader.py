from .coco import build_coco_dsets
from .vg import build_vg_dsets
import mindspore.communication.management as dist
from mindspore.dataset import DistributedSampler, GeneratorDataset


def build_loaders(cfg, mode='train'):
    assert mode in ['train', 'val', 'test']
    params = cfg.data.parameters

    if cfg.data.type == 'COCO-stuff':
        dataset = build_coco_dsets(cfg, mode=mode)
    elif cfg.data.type == 'VG':
        dataset = build_vg_dsets(cfg, mode=mode)
    else:
        raise NotImplementedError

    is_distributed = False
    if dist.get_group_size() > 1:
        is_distributed = True

    loader_kwargs = {
        'column_names': ["image", "instance"],
        'num_parallel_workers': params.loader_num_workers,
        'shuffle': params[mode].shuffle if not is_distributed else False,
    }
    rank = dist.get_rank()
    group_size = dist.get_group_size()
    if is_distributed:
        if mode == 'train':
            sampler = DistributedSampler(group_size, rank)
        else:
            sampler = DistributedSampler(group_size, rank, shuffle=False)
        loader_kwargs['sampler'] = sampler
        del loader_kwargs['shuffle']
    data_loader = GeneratorDataset(dataset, **loader_kwargs)
    data_loader = data_loader.batch(batch_size=params[mode].batch_size)

    return data_loader
