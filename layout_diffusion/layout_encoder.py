"""
Transformer implementation adapted from CLIP ViT:
https://github.com/openai/CLIP/blob/4c0275784d6d9da97ca1f47eaaee31de1867da91/clip/model.py
"""

import math


import mindspore
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore import numpy as np
from mindspore.common.initializer import initializer


def xf_convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Dense, nn.Conv2d, nn.Conv2dTranspose)):
        l.weight = ms.Parameter(l.weight.data.half(), l.weight.name)
        if l.bias is not None:
            l.bias = ms.Parameter(l.bias.data.half(), l.bias.name)


class LayerNorm(nn.LayerNorm):
    """
    Implementation that supports fp16 inputs but fp32 gains/biases.
    """

    def construct(self, x: Tensor):
        return super().construct(x.float()).to(x.dtype)


class MLP(nn.Cell):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.c_fc = nn.Dense(width, width * 4, weight_init="Uniform", bias_init="Uniform")
        self.c_proj = nn.Dense(width * 4, width, weight_init="Uniform", bias_init="Uniform")
        self.gelu = nn.GELU()

    def construct(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Cell):
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def construct(self, qkv, key_padding_mask=None):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.n_heads // 3
        scale = 1 / np.sqrt(np.sqrt(attn_ch, dtype=qkv.dtype), dtype=qkv.dtype)
        qkv = qkv.view(bs, n_ctx, self.n_heads, -1)
        q, k, v = ops.split(qkv, attn_ch, axis=-1)

        weight = ops.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterward

        if key_padding_mask is not None:
            weight = weight.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # (N, 1, 1, L1)
                float('-inf'),
            )
        wdtype = weight.dtype
        weight = ops.softmax(weight.float(), axis=-1).to(wdtype)
        return ops.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class MultiheadAttention(nn.Cell):
    def __init__(self, width, heads):
        super().__init__()
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Dense(width, width * 3, weight_init="Uniform", bias_init="Uniform")
        self.c_proj = nn.Dense(width, width, weight_init="Uniform", bias_init="Uniform")
        self.attention = QKVMultiheadAttention(heads)

    def construct(self, x, key_padding_mask=None):
        x = self.c_qkv(x)
        x = self.attention(x, key_padding_mask)
        x = self.c_proj(x)
        return x


class ResidualAttentionBlock(nn.Cell):
    def __init__(
            self,
            width: int,
            heads: int,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            width,
            heads,
        )
        self.ln_1 = LayerNorm((width,), epsilon=1e-5)
        self.mlp = MLP(width)
        self.ln_2 = LayerNorm((width,), epsilon=1e-5)

    def construct(self, x: ms.Tensor, key_padding_mask=None):
        x = x + self.attn(self.ln_1(x), key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Cell):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.CellList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                )
                for _ in range(layers)
            ]
        )

    def construct(self, x: ms.Tensor, key_padding_mask=None):
        for block in self.resblocks:
            x = block(x, key_padding_mask)
        return x


class LayoutTransformerEncoder(nn.Cell):
    def __init__(
            self,
            layout_length: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            num_heads: int,
            use_final_ln: bool,
            num_classes_for_layout_object: int,
            mask_size_for_layout_object: int,
            used_condition_types=None,
            use_positional_embedding=True,
            resolution_to_attention=[],
            use_key_padding_mask=False,
            not_use_layout_fusion_module=False
    ):
        super().__init__()
        if used_condition_types is None:
            used_condition_types = ['obj_class', 'obj_bbox', 'obj_mask']
        self.not_use_layout_fusion_module = not_use_layout_fusion_module
        self.use_key_padding_mask = use_key_padding_mask
        self.used_condition_types = []
        for i in used_condition_types:
            self.used_condition_types.append(i)
        self.num_classes_for_layout_object = num_classes_for_layout_object
        self.mask_size_for_layout_object = mask_size_for_layout_object
        if not self.not_use_layout_fusion_module:
            self.transform = Transformer(
                width=hidden_dim,
                layers=num_layers,
                heads=num_heads
            )
        self.use_positional_embedding = use_positional_embedding
        if self.use_positional_embedding:
            self.positional_embedding = ms.Parameter(ms.numpy.empty(layout_length, hidden_dim))
        self.transformer_proj = nn.Dense(hidden_dim, output_dim, weight_init="Uniform", bias_init="Uniform")

        if 'obj_class' in self.used_condition_types:
            self.obj_class_embedding = nn.Embedding(num_classes_for_layout_object, hidden_dim)
        if 'obj_bbox' in self.used_condition_types:
            self.obj_bbox_embedding = nn.Dense(4, hidden_dim, weight_init="Uniform", bias_init="Uniform")
        if 'obj_mask' in self.used_condition_types:
            self.obj_mask_embedding = nn.Dense(mask_size_for_layout_object * mask_size_for_layout_object, hidden_dim, weight_init="Uniform", bias_init="Uniform")

        if use_final_ln:
            self.final_ln = LayerNorm((hidden_dim,), epsilon=1e-5)
        else:
            self.final_ln = None

        self.dtype = ms.float32

        self.resolution_to_attention = []
        for i in resolution_to_attention:
            self.resolution_to_attention.append(i)

        self.image_patch_bbox_embedding = {}
        for resolution in self.resolution_to_attention:
            interval = 1.0 / resolution
            self.image_patch_bbox_embedding['resolution{}'.format(resolution)] = ms.Tensor(
                [(interval * j, interval * i, interval * (j + 1), interval * (i + 1)) for i in range(resolution) for j
                 in range(resolution)],
            )  # (L, 4)

    def convert_to_fp16(self):
        self.dtype = ms.float16
        if not self.not_use_layout_fusion_module:
            self.transform.apply(xf_convert_module_to_f16)
        self.transformer_proj.to_float(ms.float16)
        if self.use_positional_embedding:
            self.positional_embedding.to_float(ms.float16)
        if 'obj_class' in self.used_condition_types:
            self.obj_class_embedding.to_float(ms.float16)
        if 'obj_bbox' in self.used_condition_types:
            self.obj_bbox_embedding.to_float(ms.float16)
        if 'obj_mask' in self.used_condition_types:
            self.obj_mask_embedding.to_float(ms.float16)

    def construct(self, obj_class=None, obj_bbox=None, is_valid_obj=None, obj_mask=None, image_patch_bbox=None):
        assert (obj_class is not None) or (obj_bbox is not None) or (obj_mask is not None)
        outputs = {}

        xf_in = None
        if self.use_positional_embedding:
            xf_in = self.positional_embedding[None]

        if 'obj_class' in self.used_condition_types:
            obj_class_embedding = self.obj_class_embedding(obj_class.long())
            if xf_in is None:
                xf_in = obj_class_embedding
            else:
                xf_in = xf_in + obj_class_embedding
            outputs['obj_class_embedding'] = obj_class_embedding.permute(0, 2, 1)

        if 'obj_bbox' in self.used_condition_types:
            obj_bbox_embedding = self.obj_bbox_embedding(obj_bbox.to(self.dtype))
            if xf_in is None:
                xf_in = obj_bbox_embedding
            else:
                xf_in = xf_in + obj_bbox_embedding
            outputs['obj_bbox_embedding'] = obj_bbox_embedding.permute(0, 2, 1)
            for resolution in self.resolution_to_attention:
                outputs['image_patch_bbox_embedding_for_resolution{}'.format(resolution)] = ops.repeat_interleave(
                    input=self.obj_bbox_embedding(
                        self.image_patch_bbox_embedding['resolution{}'.format(resolution)].to(self.dtype)
                    ).unsqueeze(0),
                    repeats=obj_bbox_embedding.shape[0],
                    axis=0
                ).permute(0, 2, 1)

        if 'obj_mask' in self.used_condition_types:
            if xf_in is None:
                xf_in = self.obj_mask_embedding(obj_mask.view(*obj_mask.shape[:2], -1).to(self.dtype))
            else:
                xf_in = xf_in + self.obj_mask_embedding(obj_mask.view(*obj_mask.shape[:2], -1).to(self.dtype))

        if 'is_valid_obj' in self.used_condition_types:
            outputs['key_padding_mask'] = (1 - is_valid_obj).bool()  # (N, L2)

        key_padding_mask = outputs['key_padding_mask'] if self.use_key_padding_mask else None
        if self.not_use_layout_fusion_module:
            xf_out = xf_in.to(self.dtype)
        else:
            xf_out = self.transform(xf_in.to(self.dtype), key_padding_mask)  # NLC

        if self.final_ln is not None:
            xf_out = self.final_ln(xf_out)
        xf_proj = self.transformer_proj(xf_out[:, 0])  # NC
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL

        outputs['xf_proj'] = xf_proj
        outputs['xf_out'] = xf_out

        return outputs
