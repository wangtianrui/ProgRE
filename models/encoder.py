# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
# 2022.07 - Modified the code to support Mindspore
#           Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder definition."""

from typing import Tuple, Optional

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.initializer import Constant, Normal, initializer, Uniform

from .layernorm import LayerNorm
from .attention import MultiheadAttention
from .embedding import Wav2Vec2ConvPositionEncoding
from .encoder_layer import TransformerEncoderLayer
from .dropout import Dropout

def index_put(tensor, indices, value):
    """input values into tensor using index."""
    tensor = ops.cast(tensor, mstype.float32)
    for _ in range(indices.ndim, tensor.ndim):
        indices = indices.expand_dims(-1)
    if indices.shape[-1] < tensor.shape[-1]:
        indices = indices.expand_as(tensor)
    tensor = ops.mul(tensor, ~indices) + ops.mul(value, indices)

    return tensor

class TransformerSentenceEncoderLayer(nn.Cell):
    """TransformerSentenceEncoderLayer."""

    def __init__(self,
                 embedding_dim: float = 768,
                 ffn_embedding_dim: float = 3072,
                 num_attention_heads: int = 8,
                 dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 activation_dropout_rate: float = 0.1,
                 activation_fn: str = 'relu',
                 layer_norm_first: bool = False,
                 compute_type=ms.float32) -> None:
        super(TransformerSentenceEncoderLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.activation_dropout_rate = activation_dropout_rate

        if activation_fn == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation_fn == 'gelu':
            self.activation_fn = nn.GELU(approximate=False)
        else:
            raise NotImplementedError('{} not implement yet.')

        self.self_attn = MultiheadAttention(self.embedding_dim,
                                            num_attention_heads,
                                            dropout_rate=attention_dropout_rate)

        self.drupout1 = Dropout(1 - dropout_rate)
        self.dropout2 = Dropout(1 - self.activation_dropout_rate)
        self.dropout3 = Dropout(1 - dropout_rate)

        self.layer_norm_first = layer_norm_first

        self.self_attn_layer_norm = nn.LayerNorm((self.embedding_dim,), epsilon=1e-5).to_float(mstype.float32)
        self.fc1 = nn.Dense(self.embedding_dim, ffn_embedding_dim).to_float(compute_type)
        self.fc2 = nn.Dense(ffn_embedding_dim, self.embedding_dim).to_float(compute_type)

        self.final_layer_norm = nn.LayerNorm((self.embedding_dim,), epsilon=1e-5).to_float(mstype.float32)

    def construct(self, x, self_attn_mask=None, self_attn_padding_mask=None):
        """Construct."""
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x = self.self_attn(query=x,
                               key=x,
                               value=x,
                               key_padding_mask=self_attn_padding_mask,
                               attn_mask=self_attn_mask)
            x = self.drupout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            x = self.dropout3(x)
            x = residual + x
        else:
            x = self.self_attn(query=x,
                               key=x,
                               value=x,
                               key_padding_mask=self_attn_padding_mask,
                               attn_mask=self_attn_mask)

            x = self.drupout1(x)
            x = x + residual

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            x = self.dropout3(x)
            x = x + residual
            x = self.final_layer_norm(x)

        return x


class Wav2VecTransformerEncoder(nn.Cell):
    """Wav2VecTransformerEncoder."""

    def __init__(self,
                 encoder_embed_dim: int = 768,
                 encoder_ffn_embed_dim: int = 3072,
                 encoder_num_attention_heads: int = 8,
                 num_encoder_layers: int = 8,
                 conv_pos_kernel_size: int = 128,
                 conv_pos_groups: int = 16,
                 conv_pos_use_same_pad: bool = True,
                 pos_conv_depth: int = 1,
                 dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 activation_dropout_rate: float = 0.1,
                 activation_fn: str = 'relu',
                 layer_norm_first: bool = False,
                 required_seq_len_multiple: float = 2,
                 encoder_layer_drop_rate: float = 0.05,
                 compute_type=mstype.float32):
        super(Wav2VecTransformerEncoder, self).__init__()
        self.dropout_rate = dropout_rate
        self.embedding_dim = encoder_embed_dim
        self.required_seq_len_multiple = required_seq_len_multiple

        if pos_conv_depth > 1:
            # TODO: Multilayer position conv encoding
            pass
        else:
            self.pos_conv = Wav2Vec2ConvPositionEncoding(self.embedding_dim,
                                                         conv_pos_kernel_size,
                                                         conv_pos_groups,
                                                         use_same_pad=conv_pos_use_same_pad,
                                                         compute_type=compute_type)

        self.layers = nn.CellList([
            TransformerSentenceEncoderLayer(embedding_dim=self.embedding_dim,
                                            ffn_embedding_dim=encoder_ffn_embed_dim,
                                            num_attention_heads=encoder_num_attention_heads,
                                            dropout_rate=dropout_rate,
                                            attention_dropout_rate=attention_dropout_rate,
                                            activation_dropout_rate=activation_dropout_rate,
                                            activation_fn=activation_fn,
                                            layer_norm_first=layer_norm_first,
                                            compute_type=compute_type) for _ in range(num_encoder_layers)
        ])

        self.layer_norm_first = layer_norm_first
        self.layer_norm = nn.LayerNorm((self.embedding_dim,), epsilon=1e-5).to_float(mstype.float32)

        self.dropout = Dropout(1 - self.dropout_rate)

        self.layer_drop_rate = encoder_layer_drop_rate
        # TODO: Implement layer drop.

        self.init_params()

    def init_params(self):
        """Init params."""

        # TODO: Extraction as a public method
        def normal_(weight: ms.Tensor):
            weight.set_data(initializer(Normal(sigma=0.02, mean=0.0), weight.shape, weight.dtype))

        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                normal_(cell.weight)
                if cell.has_bias:
                    cell.bias.set_data(initializer(Constant(0), cell.bias.shape, cell.bias.dtype))
            # TODO: Embedding
            if isinstance(cell, MultiheadAttention):
                normal_(cell.q_proj.weight)
                normal_(cell.k_proj.weight)
                normal_(cell.v_proj.weight)

    @staticmethod
    def index_put(tensor, indices, value):
        # TODO: Extraction as a public method
        for _ in range(indices.ndim, tensor.ndim):
            indices = indices.expand_dims(-1)
        if indices.shape[-1] < tensor.shape[-1]:
            indices = indices.expand_as(tensor)
        tensor = ops.mul(tensor, ~indices) + ops.mul(value, indices)

        return tensor

    # TODO: implement pad_bool_to_multiple

    def extract_feature(self, x, padding_mask=None, output_layer=None):
        """Extract feature."""
        if padding_mask is not None:
            x = self.index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.swapaxes(1, 2))
        x_conv = x_conv.swapaxes(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # TODO: implement pad_to_multiple
        # x, pad_length = self.pad_to_multiple(x, self.required_seq_len_multiple, dim=-2)
        # if pad_length > 0 and padding_mask is None:
        #     padding_mask = ops.Zeros()((x.shape[0], x.shape[1]), ms.bool_)
        #     padding_mask[:, -pad_length:] = True
        # else:
        results = [x.copy()]

        x = self.dropout(x)

        # BxTxC -> TxBxC
        x = x.swapaxes(0, 1)

        # TODO: Implement layer drop
        for i, layer in enumerate(self.layers):
            
            x = layer(x, self_attn_padding_mask=padding_mask)
            if i == output_layer:
                break
            results.append(x.swapaxes(0, 1))

        # TxBxC -> BxTxC
        x = x.swapaxes(0, 1)

        return x, results

    def construct(self, x, padding_mask=None, output_layer=None):
        x, results = self.extract_feature(x, padding_mask, output_layer)

        return x, results
