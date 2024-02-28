"""
Train a diffusion model on images.
"""

import argparse
import sys

sys.path.append('/home/yangzheng/layout-diffusion-mindspore')
import mindspore as ms
from mindspore import context
from mindspore.communication import management as dist
from omegaconf import OmegaConf

from layout_diffusion import logger
from layout_diffusion.train_util import TrainLoop
from layout_diffusion.util import loopy
from layout_diffusion.layout_diffusion_unet import build_model
from layout_diffusion.resample import build_schedule_sampler
from layout_diffusion.dataset.data_loader import build_loaders
from layout_diffusion.respace import build_diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--config_file", type=str, default='/home/yangzheng/layout-diffusion-mindspore/configs/VG_128x128/LayoutDiffusion_large.yaml')
    known_args, unknown_args = parser.parse_known_args()

    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)
    # print(OmegaConf.to_yaml(cfg))

    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(device_target="GPU")
    dist.init(backend_name="nccl")
    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                 gradients_mean=True,
                                 device_num=dist.get_group_size(),
                                 parameter_broadcast=True)

    # dist_util.setup_dist(local_rank=cfg.local_rank)
    logger.configure(dir=cfg.train.log_dir)
    logger.log('current rank = {}, total_num = {}, \n'.format(dist.get_rank(), dist.get_group_size()))

    logger.log("creating model...")
    model = build_model(cfg)
    # print(model)

    logger.log("creating diffusion...")
    diffusion = build_diffusion(cfg)

    logger.log("creating schedule sampler...")
    schedule_sampler = build_schedule_sampler(cfg, diffusion)

    logger.log("creating data loader...")
    train_loader = build_loaders(cfg, mode='train')
    logger.log("training...")
    trainer = TrainLoop(
        model=model,
        diffusion=diffusion,
        schedule_sampler=schedule_sampler,
        data=loopy(train_loader),
        batch_size=cfg.data.parameters.train.batch_size,
        **cfg.train
    )
    trainer.run_loop()


if __name__ == "__main__":
    main()
