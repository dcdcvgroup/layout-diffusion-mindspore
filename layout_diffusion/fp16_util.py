"""
Helpers to train with 16-bit precision.
"""

import numpy as np
import mindspore as ms
from mindspore import nn, ops, jit_class

from . import logger

INITIAL_LOG_LOSS_SCALE = 20.0


def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        # return tensors[0].contiguous().view(-1)
        return ops.reshape(tensors[0], (-1,))
    # flat = ops.cat([t.contiguous().view(-1) for t in tensors], axis=0)
    flat = ops.concat([ops.reshape(t, (-1,)) for t in tensors], axis=0)
    return flat


def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight = ms.Parameter(l.weight.data.half(), l.weight.name)
        if l.bias is not None:
            l.bias = ms.Parameter(l.bias.data.half(), l.bias.name)


def all_convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Embedding, nn.Dense, nn.Conv2dTranspose, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight = ms.Parameter(l.weight.data.half(), l.weight.name)
        if l.bias is not None:
            l.bias = ms.Parameter(l.bias.data.half(), l.bias.name)


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight = ms.Parameter(l.weight.data.float(), l.weight.name)
        if l.bias is not None:
            l.bias = ms.Parameter(l.bias.data.float(), l.bias.name)


def make_master_params(param_groups_and_shapes):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    master_params = []
    # for param_group, shape in param_groups_and_shapes:
    #     master_param = ms.Parameter(
    #         _flatten_dense_tensors(
    #             [param.float() for (_, param) in param_group]
    #         ).view(shape)
    #     )
    #     master_param.requires_grad = True
    #     master_params.append(master_param)
    return master_params


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)


def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])


def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(p.name, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(p.name, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]


def master_params_to_param_dict(
        model, master_params
):

    param_dict = model.parameters_dict()
    for i, (name, _value) in enumerate(model.parameters_dict().items()):
        assert name in param_dict
        param_dict[name] = master_params[i]
    return param_dict


def param_dict_to_master_params(model, param_dict):
    master_params = [param_dict[name] for name, _ in model.parameters_dict().items()]
    return master_params


def param_dict_to_param_list(param_dict):
    param_list = []
    temp_list = list(param_dict.items())
    for (name, data) in temp_list:
        param_list.append({"name": name, "data": data})
    return param_list


class MixedPrecisionTrainer:
    def __init__(
            self,
            *,
            model,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            only_update_parameters_that_require_grad=False
    ):
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth

        if self.use_fp16:
            self.model.convert_to_fp16()
        self.only_update_parameters_that_require_grad = only_update_parameters_that_require_grad
        if only_update_parameters_that_require_grad:
            self.model_params = self.get_parameters_that_require_grad()
        else:
            self.model_params = list(self.model.get_parameters())

        self.master_params = self.model_params

    def get_named_parameters_that_require_grad(self):
        named_parameters_that_require_grad = []
        for (name, parameter) in list(self.model.parameters_dict().items()):
            if parameter.requires_grad:
                named_parameters_that_require_grad.append((name, parameter))

        return named_parameters_that_require_grad

    def get_parameters_that_require_grad(self):
        parameters_that_require_grad = []
        for (name, parameter) in list(self.model.parameters_dict().items()):
            if parameter.requires_grad:
                parameters_that_require_grad.append(parameter)

        return parameters_that_require_grad

    def master_params_to_param_dict(self, master_params):
        return master_params_to_param_dict(
            self.model, master_params
        )

    def param_dict_to_master_params(self, param_dict):
        return param_dict_to_master_params(self.model, param_dict)


@jit_class
class Accumulator():
    def __init__(self, optimizer, accumulate_step, clip_norm=1.0):
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.inner_grads = optimizer.parameters.clone(prefix="accumulate_", init='zeros')
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init='zeros')
        self.counter = ms.Parameter(ms.Tensor(1, ms.int32), 'counter_')
        assert accumulate_step > 0
        self.accumulate_step = accumulate_step
        self.map = ops.HyperMap()

    def __call__(self, grads):
        # 将单步获得的梯度累加至Accumulator的inner_grads
        self.map(ops.partial(ops.assign_add), self.inner_grads, grads)
        if self.counter % self.accumulate_step == 0:
            # 如果达到累积步数，进行参数优化更新
            self.optimizer(self.inner_grads)
            # 完成参数优化更新后，清零inner_grads
            self.map(ops.partial(ops.assign), self.inner_grads, self.zeros)
        # 计算步数加一
        ops.assign_add(self.counter, ms.Tensor(1, ms.int32))

        return True


def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)
