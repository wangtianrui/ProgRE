# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Weight norm conv1d."""

import mindspore as ms
from mindspore import Parameter
from mindspore import log as logger
from mindspore import ops
from mindspore._checkparam import GE, check_value_type, check_int, check_non_negative_int, check_positive_int, check_string, check_bool
from mindspore.common.initializer import HeNormal, initializer
from mindspore.nn import Cell
from mindspore.ops.primitive import constexpr


@constexpr
def _check_input_3d(input_shape, op_name):
    if len(input_shape) != 3:
        raise ValueError(f"For '{op_name}', the dimension of input must be 3d, but got {len(input_shape)}.")


class WeightNormConv1d(Cell):
    """Weight norm conv1d.

    Args:
        in_channels (int): channels of input tensor
        out_channels (int): channels of output tensor
        kernel_size (int): kernel size of conv
        stride (int): stride of conv. default: 1
        pad_mode (str): specifies padding mode. the optional values are
            "same", "valid" and "pad". default: "same".
        padding (int): padding number of conv. default: 0
        dilation (int): dilation conv space. default: 1, means no dilation
        group (int): group num. will divide channels into groups to calculate.
            default: 1, means no group divided
        has_bias (bool): whether conv has bias parameter. default: False,
        bias_init (str): specifies bias init method. the optional values are
            "normal", "ones" or "zeros", etc. default: "zeros"
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 bias_init='zeros'):
        super(WeightNormConv1d, self).__init__()

        # TODO: Implement weight_init

        check_value_type('kernel_size', kernel_size, [int], self.cls_name)
        check_value_type('stride', stride, [int], self.cls_name)
        check_value_type('padding', padding, [int], self.cls_name)
        check_value_type('dilation', dilation, [int], self.cls_name)
        check_int(kernel_size, 1, GE, 'kernel_size', self.cls_name)
        check_int(stride, 1, GE, 'stride', self.cls_name)
        check_non_negative_int(padding, 'padding', self.cls_name)
        check_int(dilation, 1, GE, 'dilation', self.cls_name)

        self.in_channels = check_positive_int(in_channels, 'in_channels', self.cls_name)
        self.out_channels = check_positive_int(out_channels, 'out_channels', self.cls_name)
        self.kernel_size = (1, kernel_size)
        self.stride = (1, stride)
        self.pad_mode = pad_mode
        self.dilation = (1, dilation)
        self.group = check_positive_int(group)
        self.has_bias = has_bias
        self.bias_init = bias_init

        if isinstance(padding, int):
            check_non_negative_int(padding, 'padding', self.cls_name)
        elif isinstance(padding, tuple):
            for pad in padding:
                check_non_negative_int(pad, 'padding item', self.cls_name)
        else:
            raise TypeError(f"For '{self.cls_name}', the type of 'padding' must be int or tuple(int), "
                            f'but got {type(padding).__name__}.')

        for kernel_size_elem in self.kernel_size:
            check_positive_int(kernel_size_elem, 'kernel_size item', self.cls_name)
        for stride_elem in self.stride:
            check_positive_int(stride_elem, 'stride item', self.cls_name)
        for dilation_elem in self.dilation:
            check_positive_int(dilation_elem, 'dilation item', self.cls_name)
        if in_channels % group != 0:
            raise ValueError(f"For '{self.cls_name}', the attr 'in_channels' must be divisible by attr 'group', "
                             f"but got 'in_channels': {in_channels} and 'group': {group}.")
        if out_channels % group != 0:
            raise ValueError(f"For '{self.cls_name}', the 'out_channels' must be divisible by attr 'group', "
                             f"but got 'out_channels': {out_channels} and 'group': {group}.")

        self.padding = (0, 0, padding, padding)
        check_string(pad_mode, ['valid', 'same', 'pad'], 'pad_mode', self.cls_name)

        self.weight_shape = (out_channels, in_channels // group, *self.kernel_size)
        weight = initializer(HeNormal(), self.weight_shape, ms.float32)
        self.weight_g = Parameter(ops.LpNorm((0, 1, 2), keep_dims=True)(weight), name='weight_g')
        self.weight_v = Parameter(weight / self.weight_g, name='weight_v')

        if check_bool(has_bias, 'has_bias', self.cls_name):
            self.bias = Parameter(initializer(self.bias_init, [out_channels], ms.float16), name='bias')
        else:
            if self.bias_init != 'zeros':
                logger.warning("Value of 'has_bias' is False, value of 'bias_init' will be ignored.")
            self.bias = None

        self.conv2d = ops.Conv2D(out_channel=self.out_channels,
                                 kernel_size=self.kernel_size,
                                 mode=1,
                                 pad_mode=self.pad_mode,
                                 pad=self.padding,
                                 stride=self.stride,
                                 dilation=self.dilation,
                                 group=self.group)

    def reset_weight(self, weight):
        """reset weight."""
        assert isinstance(weight, ms.Tensor), 'weight should be type of Tensor.'
        assert weight.shape == self.weight_shape

        weight = ops.cast(weight, ms.float32)
        weight_g = ops.LpNorm((0, 1, 2), keep_dims=True)(weight)
        weight_v = weight / weight_g
        weight_g = ops.cast(weight_g, self.weight_g.dtype)
        weight_v = ops.cast(weight_v, self.weight_v.dtype)
        self.weight_g.set_data(ms.Tensor(weight_g))
        self.weight_v.set_data(ms.Tensor(weight_v))

    def construct(self, x):
        """Weight norm forward."""
        weight = self.weight_v * (self.weight_g / ops.LpNorm((0, 1, 2), keep_dims=True)(self.weight_v))
        weight = ops.cast(weight, ms.float16)

        x_shape = x.shape
        _check_input_3d(x_shape, self.cls_name)
        x = x.expand_dims(2)
        x = ops.cast(x, ms.float16)
        output = self.conv2d(x, weight)
        if self.has_bias:
            output = ops.BiasAdd()(output, self.bias)

        output = output.squeeze(2)
        return output
