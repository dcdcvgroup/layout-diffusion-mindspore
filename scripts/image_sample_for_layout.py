"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
import json
import time
import sys

sys.path.append('/home/yangzheng/layout-diffusion-mindspore')

import numpy as np
import imageio
import mindspore
import mindspore as ms
from mindspore import context, ops, Tensor
from mindspore.ops import stop_gradient
from mindspore.communication import management as dist  # mindspore distributed module

from omegaconf import OmegaConf

from layout_diffusion import logger
from layout_diffusion.dataset.data_loader import build_loaders
from layout_diffusion.layout_diffusion_unet import build_model
from layout_diffusion.respace import build_diffusion
from layout_diffusion.util import fix_seed
from layout_diffusion.dataset.util import image_unnormalize_batch
from dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver


def imageio_save_image(img_tensor, path):
    '''
    :param img_tensor: (C, H, W) ms.Tensor
    :param path:
    :return:
    '''
    tmp_img = image_unnormalize_batch(img_tensor).clamp(0.0, 1.0)
    imageio.imsave(
        uri=path,
        im=(tmp_img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8),  # (H, W, C) numpy
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--config_file", type=str,
                        default='/home/yangzheng/layout-diffusion-mindspore/configs/VG_128x128/LayoutDiffusion_large.yaml')

    known_args, unknown_args = parser.parse_known_args()

    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)

    # dist_util.setup_dist(local_rank=cfg.local_rank)

    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(device_target="GPU")
    dist.init()
    logger.log("setting context...")

    data_loader = build_loaders(cfg, mode='test')

    total_num_batches = len(data_loader)
    log_dir = os.path.join(cfg.sample.log_root, 'conditional_{}'.format(cfg.sample.timestep_respacing),
                           'sample{}x{}'.format(total_num_batches, int(cfg.sample.sample_times)),
                           cfg.sample.sample_suffix)
    logger.configure(dir=log_dir)
    # logger.log('current rank == {}, total_num = {}, \n, {}'.format(dist.get_rank(), dist.get_world_size(), cfg))
    logger.log(OmegaConf.to_yaml(cfg))

    logger.log("creating model...")
    model = build_model(cfg)
    # logger.log(model)

    if cfg.sample.fix_seed:
        fix_seed()

    if cfg.sample.pretrained_model_path:
        logger.log("loading model from {}".format(cfg.sample.pretrained_model_path))
        checkpoint = ms.load_checkpoint(cfg.sample.pretrained_model_path)
        # if 'layout_encoder.obj_box_embedding.weight' in list(checkpoint.keys()):
        #     checkpoint['layout_encoder.obj_bbox_embedding.weight'] = checkpoint.pop(
        #         'layout_encoder.obj_box_embedding.weight')
        #     checkpoint['layout_encoder.obj_bbox_embedding.bias'] = checkpoint.pop(
        #         'layout_encoder.obj_box_embedding.bias')
        try:
            ms.load_param_into_net(model, checkpoint, strict_load=True)
            logger.log('successfully load the entire model')
        except:
            logger.log('not successfully load the entire model, try to load part of model')
            ms.load_param_into_net(model, checkpoint, strict_load=False)

    if cfg.sample.use_fp16:
        model.convert_to_fp16()
        # model.to_float(ms.float16)

    def model_fn(x, t, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, **kwargs):
        assert obj_class is not None
        assert obj_bbox is not None
        cond_image, cond_extra_outputs = model(
            x, t,
            obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
            is_valid_obj=is_valid_obj
        )
        cond_mean, cond_variance = ops.chunk(cond_image, 2, axis=1)

        obj_class = ops.ones_like(obj_class).fill(model.layout_encoder.num_classes_for_layout_object - 1)
        obj_class[:, 0] = 0

        obj_bbox = ops.zeros_like(obj_bbox)
        obj_bbox[:, 0] = Tensor([0, 0, 1, 1]).astype(ms.float32)

        is_valid_obj = ops.zeros_like(obj_class)
        is_valid_obj[:, 0] = 1.0

        if obj_mask is not None:
            obj_mask = ops.zeros_like(obj_mask)
            obj_mask[:, 0] = ops.ones(obj_mask.shape[-2:])

        uncond_image, uncond_extra_outputs = model(
            x, t,
            obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
            is_valid_obj=is_valid_obj
        )
        uncond_mean, uncond_variance = ops.chunk(uncond_image, 2, axis=1)

        mean = cond_mean + cfg.sample.classifier_free_scale * (cond_mean - uncond_mean)

        if cfg.sample.sample_method in ['ddpm', 'ddim']:
            return [ops.cat([mean, cond_variance], axis=1), cond_extra_outputs]
        else:
            return mean

    dir_names = ['generated_imgs', 'real_imgs', 'gt_annotations']
    if cfg.sample.save_cropped_images:
        dir_names.extend(['generated_cropped_imgs', 'real_cropped_imgs'])
    if cfg.sample.save_images_with_bboxs:
        dir_names.extend(['real_imgs_with_bboxs', 'generated_imgs_with_bboxs', 'generated_images_with_each_bbox'])
    if cfg.sample.save_sequence_of_obj_imgs:
        dir_names.extend(['obj_imgs_from_unresized_gt_imgs', 'obj_imgs_from_resized_gt_imgs'])

    for dir_name in dir_names:
        os.makedirs(os.path.join(log_dir, dir_name), exist_ok=True)

    if cfg.sample.save_cropped_images:
        if cfg.data.type == 'COCO-stuff':
            for class_id in range(1, 183):  # 1-182
                if class_id not in [12, 183, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]:
                    os.makedirs(os.path.join(log_dir, 'generated_cropped_imgs', str(class_id)), exist_ok=True)
                    os.makedirs(os.path.join(log_dir, 'real_cropped_imgs', str(class_id)), exist_ok=True)
        elif cfg.data.type == 'VG':
            for class_id in range(1, 179):  # 1-178
                os.makedirs(os.path.join(log_dir, 'generated_cropped_imgs', str(class_id)), exist_ok=True)
                os.makedirs(os.path.join(log_dir, 'real_cropped_imgs', str(class_id)), exist_ok=True)
        else:
            raise NotImplementedError

    logger.log("creating diffusion...")
    if cfg.sample.sample_method == 'dpm_solver':
        noise_schedule = NoiseScheduleVP(schedule='linear')
    elif cfg.sample.sample_method in ['ddpm', 'ddim']:
        diffusion = build_diffusion(cfg, timestep_respacing=cfg.sample.timestep_respacing)
    else:
        raise NotImplementedError

    logger.log('sample method = {}'.format(cfg.sample.sample_method))
    logger.log("sampling...")
    start_time = time.time()
    total_time = 0.0

    for batch_idx, batch in enumerate(data_loader):
        total_time += (time.time() - start_time)

        print('rank={}, batch_id={}'.format(dist.get_rank(), batch_idx))

        imgs, cond = batch
        model_kwargs = {
            'obj_class': cond['obj_class'],
            'obj_bbox': cond['obj_bbox'],
            'is_valid_obj': cond['is_valid_obj']
        }
        if 'obj_mask' in cfg.data.parameters.used_condition_types:
            model_kwargs['obj_mask']: cond['obj_mask']

        for sample_idx in range(cfg.sample.sample_times):
            start_time = time.time()
            if cfg.sample.sample_method == 'dpm_solver':
                wrappered_model_fn = model_wrapper(
                    model_fn,
                    noise_schedule,
                    is_cond_classifier=False,
                    total_N=1000,
                    model_kwargs=model_kwargs
                )

                dpm_solver = DPM_Solver(wrappered_model_fn, noise_schedule)

                x_T = ops.randn((imgs.shape[0], 3, cfg.data.parameters.image_size, cfg.data.parameters.image_size))
                sample = dpm_solver.sample(
                    x_T,
                    steps=int(cfg.sample.timestep_respacing[0]),
                    eps=float(cfg.sample.eps),
                    adaptive_step_size=cfg.sample.adaptive_step_size,
                    fast_version=cfg.sample.fast_version,
                    clip_denoised=False,
                    rtol=cfg.sample.rtol
                )  # (B, 3, H, W)
                sample = sample.clamp(-1, 1)
            elif cfg.sample.sample_method in ['ddpm', 'ddim']:
                sample_fn = (
                    diffusion.p_sample_loop if cfg.sample.sample_method == 'ddpm' else diffusion.ddim_sample_loop)
                all_results = sample_fn(
                    model_fn, (imgs.shape[0], 3, cfg.data.parameters.image_size, cfg.data.parameters.image_size),
                    clip_denoised=cfg.sample.clip_denoised, model_kwargs=model_kwargs, cond_fn=None
                )  # (B, 3, H, W)
                last_result = all_results[-1]
                sample = last_result['sample'].clamp(-1, 1)
            else:
                raise NotImplementedError

            total_time += (time.time() - start_time)
            for img_idx in range(imgs.shape[0]):
                start_time = time.time()
                filename = cond['filename'].asnumpy()[img_idx]
                # obj_num = cond['num_obj'][img_idx]
                # obj_class = cond['obj_class'][img_idx]
                # obj_name = cond['obj_class_name'][img_idx]
                # is_valid_obj = cond['is_valid_obj'].long()[img_idx]
                # obj_bbox = cond['obj_bbox'][img_idx]
                # absolute_obj_bbox = obj_bbox.copy()
                # absolute_obj_bbox[:, 0::2] = cond[:, 0::2] * imgs[img_idx].shape[2]
                # absolute_obj_bbox[:, 1::2] = cond[:, 1::2] * imgs[img_idx].shape[1]

                # save generated imgs
                imageio_save_image(
                    img_tensor=sample[img_idx],
                    path=os.path.join(log_dir, "generated_imgs/{}_{}.png".format(filename, sample_idx)),
                )
                total_time += (time.time() - start_time)

                if sample_idx == 0:
                    # save real imgs
                    imageio_save_image(
                        img_tensor=imgs[img_idx],
                        path=os.path.join(log_dir, "real_imgs/{}.png".format(filename)),
                    )

                    # save annotations of real imgs
                    with open(os.path.join(log_dir, 'gt_annotations/{}.json'.format(filename)), 'w') as f:
                        gt_annotations = {}
                        for key, value in cond.items():
                            if isinstance(value, ms.Tensor):
                                gt_annotations[key] = value.asnumpy()[img_idx]
                            else:
                                gt_annotations[key] = value[img_idx]
                            gt_annotations[key] = gt_annotations[key].tolist()
                        f.write(json.dumps(gt_annotations))

        cur_num_batches = (batch_idx + 1) * dist.get_group_size()
        fps = (batch_idx + 1) * cfg.data.parameters.test.batch_size * cfg.sample.sample_times / (total_time)
        logger.log('FPS = {} / {} = {} imgs / second'.format(
            (batch_idx + 1) * cfg.data.parameters.test.batch_size * cfg.sample.sample_times, total_time, fps))
        logger.log(f"batch_id={batch_idx + 1} created {cur_num_batches} / {total_num_batches} batches")
        if cur_num_batches >= total_num_batches:
            break
        start_time = time.time()

    logger.log("sampling complete")


if __name__ == "__main__":
    main()
