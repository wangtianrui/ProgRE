import math
from typing import Tuple

import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.operations as ops
# pylint: disable=C0412
import numpy as np
from mindspore.common.initializer import Constant, Normal, initializer
from mindspore.common.tensor import Tensor  # pylint: disable=C0412

from .same_pad import SamePad
from .weight_norm_conv1d import WeightNormConv1d


class Wav2Vec2ConvPositionEncoding(nn.Cell):
    """Wav2VecConvPositionEncoding.

    Args:
        d_model (int): dimension of model
        kernel_size (int): kernel size of conv,
        group (int): group size of conv. default: 1
        has_bias (bool): whether has bias parameter. default: True,
        dropout_rate (float): drop rate. default: 0.0,
        use_same_pad (bool): whether to use same padding. default: True
    """

    def __init__(self,
                 d_model: int,
                 kernel_size: int,
                 group: int = 1,
                 has_bias: bool = True,
                 dropout_rate: float = 0.0,
                 use_same_pad=True,
                 compute_type=mindspore.float16):
        super(Wav2Vec2ConvPositionEncoding, self).__init__()

        std = math.sqrt((4 * (1.0-dropout_rate)) / (kernel_size*d_model))
        if use_same_pad:
            pos_conv = WeightNormConv1d(d_model,
                                        d_model,
                                        kernel_size=kernel_size,
                                        pad_mode='pad',
                                        padding=kernel_size // 2,
                                        group=group,
                                        has_bias=has_bias)
            pos_conv.reset_weight(initializer(Normal(sigma=std, mean=0), pos_conv.weight_shape,
                                              pos_conv.weight_g.dtype))
            pos_conv.bias.set_data(initializer(Constant(0), pos_conv.bias.shape, pos_conv.bias.dtype))
            self.pos_conv = nn.SequentialCell(pos_conv, SamePad(kernel_size), nn.GELU(approximate=False))
        else:
            pos_conv = WeightNormConv1d(d_model,
                                        d_model,
                                        kernel_size=kernel_size,
                                        pad_mode='same',
                                        group=group,
                                        has_bias=has_bias)
            pos_conv.reset_weight(initializer(Normal(sigma=std, mean=0), pos_conv.weight_shape,
                                              pos_conv.weight_g.dtype))
            pos_conv.bias.set_data(initializer(Constant(0), pos_conv.bias.shape, pos_conv.bias.dtype))
            self.pos_conv = nn.SequentialCell(pos_conv, nn.GELU(approximate=False))

    def construct(self, x):
        return self.pos_conv(x)
