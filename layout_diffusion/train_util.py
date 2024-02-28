import copy
import functools
import os
import time

import blobfile as bf
import mindspore
import mindspore as ms
from mindspore import ops, nn, jit
from mindspore.communication import management as dist
from mindspore.nn import AdamWeightDecay as AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer, Accumulator, param_dict_to_param_list
from layout_diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
import numpy as np
from layout_diffusion.layout_diffusion_unet import LayoutDiffusionUNetModel


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            batch_size,
            micro_batch_size,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            find_unused_parameters=False,
            only_update_parameters_that_require_grad=False,
            classifier_free=False,
            classifier_free_dropout=0.0,
            pretrained_model_path='',
            log_dir=""
    ):
        self.log_dir = log_dir
        logger.configure(dir=log_dir)
        self.model = model

        self.pretrained_model_path = pretrained_model_path
        if pretrained_model_path:
            logger.log("loading model from {}".format(pretrained_model_path))
            checkpoint = ms.load_checkpoint(pretrained_model_path)
            try:
                ms.load_param_into_net(model, checkpoint, strict_load=True)
                logger.log('successfully load the entire model')
            except:
                logger.log('not successfully load the entire model, try to load part of model')
                ms.load_param_into_net(model, checkpoint, strict_load=False)
        self.model.set_train()
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size if micro_batch_size > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 400000
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_group_size()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            only_update_parameters_that_require_grad=only_update_parameters_that_require_grad
        )

        self.opt = AdamW(
            params=model.trainable_params(), learning_rate=self.lr, eps=1e-08, weight_decay=self.weight_decay
        )

        # self.accumulate_step = self.batch_size / self.micro_batch_size
        # self.accumulator = Accumulator(self.opt, self.accumulate_step)

        self.grad_fn = ms.value_and_grad(self.compute_losses, None, self.opt.parameters)

        self.grad_reducer = ops.identity
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        reducer_flag = (parallel_mode != ms.ParallelMode.STAND_ALONE)
        if reducer_flag:
            self.grad_reducer = nn.DistributedGradReducer(self.opt.parameters, mean=True, degree=dist.get_group_size())

        if self.resume_step:
            self._load_optimizer_param()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        self.classifier_free = classifier_free
        self.classifier_free_dropout = classifier_free_dropout
        self.dropout_condition = False

    def compute_losses(self, micro, t, weights, obj_class, obj_bbox, is_valid_obj):
        mse, vb = self.diffusion.training_losses(self.model, micro, t, obj_class, obj_bbox, is_valid_obj)
        mse = (mse * weights).mean()
        vb = (vb * weights).mean()
        loss = mse + vb
        return loss, mse, vb

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"resume step = {self.resume_step}...")
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                ms.load_param_into_net(self.opt, resume_checkpoint)

        dist_util.sync_params(self.model.get_parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                param_dict = ms.load_checkpoint(ema_checkpoint)
                ema_params = self.mp_trainer.param_dict_to_master_params(param_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_param(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:07}.ckpt"
        )
        logger.log(f"try to load optimizer state from checkpoint: {opt_checkpoint}")
        if bf.exists(opt_checkpoint):
            logger.log(f"successfully loading optimizer state from checkpoint: {opt_checkpoint}")
            param_dict = ms.load_checkpoint(opt_checkpoint)
            ms.load_param_into_net(self.opt, param_dict)

    def run_loop(self):
        def run_loop_generator():
            while (
                    not self.lr_anneal_steps
                    or self.step + self.resume_step < self.lr_anneal_steps
            ):
                yield

        for _ in tqdm(run_loop_generator()):
            batch, cond = next(self.data)
            obj_class = cond['obj_class']
            if obj_class.ndim == 1:
                obj_class = obj_class.unsqueeze(0)
            obj_bbox = cond['obj_bbox']
            if obj_bbox.ndim == 2:
                obj_bbox = obj_bbox.unsqueeze(0)
            is_valid_obj = cond['is_valid_obj']
            if is_valid_obj.ndim == 1:
                is_valid_obj = is_valid_obj.unsqueeze(0)
            if self.classifier_free and self.classifier_free_dropout > 0.0:
                p = np.random.rand()
                self.dropout_condition = False
                if p < self.classifier_free_dropout:
                    self.dropout_condition = True
                    if isinstance(self.model, LayoutDiffusionUNetModel):
                        if 'obj_class' in self.model.layout_encoder.used_condition_types:
                            obj_class = ops.ones_like(obj_class).fill(
                                self.model.layout_encoder.num_classes_for_layout_object - 1)
                            obj_class[:, 0] = 0
                        if 'obj_bbox' in self.model.layout_encoder.used_condition_types:
                            obj_bbox = ops.zeros_like(obj_bbox)
                            obj_bbox[:, 0] = ops.Tensor([0., 0., 1., 1.]).astype(ms.float32)
                        is_valid_obj = ops.zeros_like(is_valid_obj)
                        is_valid_obj[:, 0] = 1.0

            self.run_step(batch, obj_class, obj_bbox, is_valid_obj)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            if self.step % self.save_interval == 0 and self.step > 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

                # if (self.step + self.resume_step) >= 100000:
                #     return

            self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, obj_class, obj_bbox, is_valid_obj):
        t, weights = self.schedule_sampler.sample(batch.shape[0])
        # for i in range(0, batch.shape[0], self.micro_batch_size):
        #     micro = batch[i: i + self.micro_batch_size]
        #     micro_class = obj_class[i: i + self.micro_batch_size]
        #     micro_bbox = obj_bbox[i: i + self.micro_batch_size]
        #     micro_valid_obj = is_valid_obj[i: i + self.micro_batch_size]
        losses, mse, vb = self.forward_backward(batch, t, weights, obj_class, obj_bbox, is_valid_obj)
        log_loss_dict(
            self.diffusion, t, {"loss": losses, "mse": mse, "vb": vb}
        )
        self._update_ema()
        self._anneal_lr()
        self.log_step()

    @jit
    def forward_backward(self, batch, t, weights, obj_class, obj_bbox, is_valid_obj):
        losses, grads = self.grad_fn(batch, t, weights, obj_class, obj_bbox, is_valid_obj)

        grads = self.grad_reducer(grads)
        self.opt(grads)
        return losses

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            for targ, src in zip(params, list(self.model.get_parameters())):
                ops.assign(targ, targ * rate + src * (1. - rate))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param in self.opt.parameters:
            param['lr']: lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            param_dict = self.mp_trainer.master_params_to_param_dict(params)
            param_list = param_dict_to_param_list(param_dict)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):07d}.ckpt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):07d}.ckpt"
                f = get_blob_logdir() + "/" + filename
                ms.save_checkpoint(param_list, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            param_list = param_dict_to_param_list(self.opt.parameters_dict())
            filename = f"opt{(self.step + self.resume_step):07d}.ckpt"
            f = get_blob_logdir() + "/" + filename
            ms.save_checkpoint(param_list, f)

        # dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.ckpt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):07d}.ckpt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().value())
        # Log the quantiles (four quartiles, in particular).
        quartile = int(4 * ts[0].value() / diffusion.num_timesteps)
        logger.logkv_mean(f"{key}_q{quartile}", values.value())
