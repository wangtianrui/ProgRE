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
"""Feature extractor for wav2vec 2.0 model."""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import os
from .conv1d import Conv1d
from .layernorm import LayerNorm
from .dropout import Dropout

class ConvLayernormGelu(nn.Cell):
    """Convolution layer with GELU activation and layer norm.

    Args:
        in_channels (int): The channel number of the input tensor of the Conv1d layer.
        out_channels (int): The channel number of the output tensor of the Conv1d layer.
        stride (int): The movement stride of the 1D convolution kernel.
        kernel_size (int): Specifies the width of the 1D convolution kernel.
        has_bias (bool): Whether the 1D convolution layer has a bias parameter.
    """

    def __init__(self, in_channels, out_channels, stride, kernel_size, has_bias=False):
        super(ConvLayernormGelu, self).__init__()
        self.layernorm = LayerNorm(out_channels)
        self.conv = Conv1d(in_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           has_bias=has_bias,
                           pad_mode="valid",
                           init="henormal")
        self.gelu = nn.GELU()
        self.transpose = ops.Transpose()

    def construct(self, x, xs_len):
        x, valid_length = self.conv(x, xs_len)
        x = self.transpose(x, (0, 2, 1))
        x = self.layernorm(x)
        x = self.transpose(x, (0, 2, 1))
        x = self.gelu(x)
        return x, valid_length


class FeatureExtractor(nn.Cell):
    """Feature extractor definition.

    Args:
        stride_list (list): List of strides for multi 1D convolution layers.
        kernel_size_list (int): List of kernels for multi 1D convolution layers.
        dim (int): The channel number of hidden representations.
        layer_norm (bool): Whether to use layer norm after each 1D convolution layer.
        feature_grad_mult (float): Scale factor for convolution gradient.
    """

    def __init__(self, stride_list, kernel_size_list, dim=512, layer_norm=False, feature_grad_mult=0.1):
        super(FeatureExtractor, self).__init__()
        self.dim = dim
        self.layer_norm = layer_norm
        # define ops
        self.conv_layers = nn.CellList()
        if layer_norm:
            conv0_ = ConvLayernormGelu(
                in_channels=1,
                out_channels=dim,
                stride=stride_list[0],
                kernel_size=kernel_size_list[0],
                has_bias=True,
            )
            self.conv_layers.append(conv0_)
            for stride, kernel_size in zip(stride_list[1:], kernel_size_list[1:]):
                conv_ = ConvLayernormGelu(
                    in_channels=dim,
                    out_channels=dim,
                    stride=stride,
                    kernel_size=kernel_size,
                    has_bias=True,
                )
                self.conv_layers.append(conv_)
        else:
            self.gnorm = nn.GroupNorm(dim, dim, eps=1e-5, affine=True)
            self.gelu = nn.GELU()
            self.expand = ops.ExpandDims()
            self.squeeze = ops.Squeeze(2)
            conv0_ = Conv1d(
                in_channel=1,
                out_channel=dim,
                stride=stride_list[0],
                kernel_size=kernel_size_list[0],
                has_bias=False,
                pad_mode="valid",
                init="henormal",
            )
            self.conv_layers.append(conv0_)
            for stride, kernel_size in zip(stride_list[1:], kernel_size_list[1:]):
                conv_ = Conv1d(
                    in_channel=dim,
                    out_channel=dim,
                    stride=stride,
                    kernel_size=kernel_size,
                    has_bias=False,
                    pad_mode="valid",
                    init="henormal",
                )
                self.conv_layers.append(conv_)

    def construct(self, x, x_len):
        """feature extractor function for wav2vec."""
        if self.layer_norm:
            for layer in self.conv_layers:
                x, x_len = layer(x, x_len)
        else:
            for i, layer in enumerate(self.conv_layers):
                x, x_len = layer(x, x_len)
                if i == 0:
                    x = self.gnorm(self.expand(x, 2))
                    x = self.squeeze(x)
                x = self.gelu(x)
        return x


class ConvFeatureExtractionBlock(nn.Cell):
    """Conv feature extraction block using self-defined Conv1d

    Args:
        in_channels (int): channels num of input tensor
        out_channels (int): channels num of output tensor
        kernel_size (int): kernel size of conv
        stride (int): stride length of conv
        has_bias (bool): whether conv has bias parameter
        dropout (float): drop rate of dropout
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, has_bias, dropout, compute_type):
        super(ConvFeatureExtractionBlock, self).__init__()

        def make_conv():
            conv = Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                has_bias=has_bias,
                pad_mode="valid",
                init="henormal",
            ).to_float(compute_type)
            return conv

        self.conv = make_conv()
        self.dropout = Dropout(1 - dropout).to_float(compute_type)
        self.gelu = nn.GELU(approximate=False).to_float(compute_type)

    def construct(self, x, x_len):
        x, valid_len = self.conv(x, x_len)
        x = self.dropout(x)
        x = self.gelu(x)
        return x, valid_len


class ConvLayerNormFeatureExtractionBlock(ConvFeatureExtractionBlock):
    """Conv layer norm feature extraction block.

    Args:
        in_channels (int): channels num of input tensor
        out_channels (int): channels num of output tensor
        kernel_size (int): kernel size of conv
        stride (int): stride length of conv
        has_bias (bool): whether conv has bias parameter
        dropout (float): drop rate of dropout
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, has_bias, dropout, compute_type):
        super(ConvLayerNormFeatureExtractionBlock, self).__init__(
            in_channels, out_channels, kernel_size, stride, has_bias, dropout, compute_type
        )
        # use self-defined layernorm
        self.layer_norm = LayerNorm((out_channels,), epsilon=1e-5).to_float(ms.float32)
        
    def construct(self, x, x_len):
        x, valid_len = self.conv(x, x_len)
        x = self.dropout(x)
        x = x.swapaxes(-2, -1)
        x = self.layer_norm(x)
        x = x.swapaxes(-2, -1)
        x = self.gelu(x)

        return x, valid_len


class ConvGroupNormFeatureExtractionBlock(ConvFeatureExtractionBlock):
    """Conv group norm feature extraction block.

    Args:
        in_channels (int): channels num of input tensor
        out_channels (int): channels num of output tensor
        kernel_size (int): kernel size of conv
        stride (int): stride length of conv
        has_bias (bool): whether conv has bias parameter
        dropout (float): drop rate of dropout
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, has_bias, dropout, compute_type):
        super(ConvGroupNormFeatureExtractionBlock, self).__init__(
            in_channels, out_channels, kernel_size, stride, has_bias, dropout, compute_type
        )

        self.group_norm = nn.GroupNorm(
            out_channels, out_channels, affine=True
        ).to_float(ms.float32)

    def construct(self, x, x_len):
        x, valid_len = self.conv(x, x_len)
        x = self.dropout(x)
        x = x.expand_dims(2)
        x = self.group_norm(x)
        x = x.squeeze(2)
        x = self.gelu(x)

        return x, valid_len


class ConvFeatureExtractionModel(nn.Cell):
    """Conv feature extract model.

    Args:
        stride_list (list): stride size list of cnn blocks
        kernel_size_list (list): kernel size list of cnn blocks
        dim (int): output dimension of conv. default 512.
        dropout (float): drop rate for dropout. default 0.0
        layer_norm (bool): whether to use layernorm after each conv.
            if set to False, only apply a GropNorm after first conv
            as described in <https://arxiv.org/abs/2006.11477>.
            default False.
        has_bias (bool): whether conv has bias parameters. default False
    """

    def __init__(self, stride_list, kernel_size_list, dim=512, dropout=0.0, layer_norm=False, has_bias=False, compute_type=ms.float16):
        super(ConvFeatureExtractionModel, self).__init__()
        self.dim = dim
        def block(in_channels, kernel_size, stride, is_layer_norm=False, is_group_norm=False):
            assert not (
                    is_layer_norm and is_group_norm
            ), "layer norm and group norm are exclusive"

            if is_layer_norm:
                return ConvLayerNormFeatureExtractionBlock(
                    in_channels, dim, kernel_size, stride, has_bias=has_bias, dropout=dropout, compute_type=compute_type
                )
            if is_group_norm:
                return ConvGroupNormFeatureExtractionBlock(
                    in_channels, dim, kernel_size, stride, has_bias=has_bias, dropout=dropout, compute_type=compute_type
                )
            return ConvFeatureExtractionBlock(
                in_channels,
                dim,
                kernel_size,
                stride,
                has_bias=has_bias,
                dropout=dropout,
                compute_type=compute_type
            )

        assert len(stride_list) == len(kernel_size_list), (
            "The lengths of stride_list and kernel_size_list should be equal."
        )

        self.kernel_size_list = kernel_size_list
        self.stride_list = stride_list
        in_channels = 1
        self.conv_layers = nn.CellList()
        for i, (kernel_size, stride) in enumerate(zip(kernel_size_list, stride_list)):
            self.conv_layers.append(
                block(
                    in_channels,
                    kernel_size,
                    stride,
                    is_layer_norm=layer_norm,
                    is_group_norm=not layer_norm and i == 0,
                )
            )
            in_channels = dim

    def cal_feat_extract_output_length(self, input_length):
        """calculate feature extract output length."""

        def _conv_output_length(input_length, kernel_size, stride):
            return ops.floor((input_length - kernel_size) / stride + 1)

        for kernel_size, stride in zip(self.kernel_size_list, self.stride_list):
            input_length = _conv_output_length(input_length, kernel_size, stride)

        return ops.cast(input_length, ms.int32)

    def construct(self, x, x_len):
        # BxT -> BxCxT
        x = x.expand_dims(1)
        for layer in self.conv_layers:
            x, x_len = layer(x, x_len)
        return x
