from typing import Tuple, Optional
import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.initializer import Constant, Normal, initializer
from .dense import Dense
from .layernorm import LayerNorm
from .embedding import Wav2Vec2ConvPositionEncoding
from .attention import MultiheadAttention
from .dropout import Dropout

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
                 layer_norm_first: bool = False) -> None:
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

        self.self_attn_layer_norm = LayerNorm((self.embedding_dim,), epsilon=1e-5).to_float(mstype.float32)
        self.fc1 = nn.Dense(self.embedding_dim, ffn_embedding_dim).to_float(mstype.float16)
        self.fc2 = nn.Dense(ffn_embedding_dim, self.embedding_dim).to_float(mstype.float16)

        self.final_layer_norm = LayerNorm((self.embedding_dim,), epsilon=1e-5).to_float(mstype.float32)

    def construct(self, x, self_attn_mask=None, self_attn_padding_mask=None):
        """Construct."""
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x = self.self_attn(query=x,
                               key=x,
                               value=x,
                               padding_mask=self_attn_padding_mask,
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
                               padding_mask=self_attn_padding_mask,
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

class FAS(nn.Cell):
    def __init__(self, in_dim, bottleneck_dim=256, global_context_att=False):
        super(FAS, self).__init__()
        self.in_dim = in_dim
        self.global_context_att = global_context_att

        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, bottleneck_dim, kernel_size=1
            )  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim,
                                 kernel_size=1)  # equals V and k in the paper
        self.out = Dense(in_dim * 2, in_dim, has_bias=True)
        self.sqrt = ops.Sqrt()
        self.layer_norm = LayerNorm((in_dim,), epsilon=1e-5).to_float(mstype.float32)
        self.transpose = ops.Transpose()
        self.expand_dims = ops.ExpandDims()

    def construct(self, x, padding_mask=None):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        mask = 1 - self.expand_dims(padding_mask, 1).astype(ms.float16)
        x = x * mask
            
        x_in = x
        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = ops.Tanh()(self.linear1(x_in))  # alpha = F.relu(self.linear1(x_in))
        alpha = ops.Softmax(axis=2)(self.linear2(alpha)*mask + (1.0-mask)*(-1000.0)) 
        mean = alpha * x
        var = alpha * (x**2) - mean**2
        std = self.sqrt(var + self.cast(ops.tuple_to_array((1e-5,)), ms.float16))
        cated = ops.Concat(axis=1)([mean, std])
        cated = self.transpose(cated, (0, 2, 1))
        return self.transpose(self.layer_norm(self.out(cated)), (0, 2, 1))

class ProgRETransformerEncoder(nn.Cell):
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
                 encoder_inter_loss_layer="3;999",
                 spk_norm=False):
        super(Wav2VecTransformerEncoder, self).__init__()
        self.dropout_rate = dropout_rate
        self.embedding_dim = encoder_embed_dim
        self.required_seq_len_multiple = required_seq_len_multiple
        self.encoder_inter_loss_layer = list(map(int, encoder_inter_loss_layer.split(";")))
        print("encoder_inter_loss_layer:"+str(self.encoder_inter_loss_layer))
        if pos_conv_depth > 1:
            # TODO: Multilayer position conv encoding
            pass
        else:
            self.pos_conv = Wav2Vec2ConvPositionEncoding(self.embedding_dim,
                                                         conv_pos_kernel_size,
                                                         conv_pos_groups,
                                                         use_same_pad=conv_pos_use_same_pad)

        self.layers = nn.CellList([
            TransformerSentenceEncoderLayer(embedding_dim=self.embedding_dim,
                                            ffn_embedding_dim=encoder_ffn_embed_dim,
                                            num_attention_heads=encoder_num_attention_heads,
                                            dropout_rate=dropout_rate,
                                            attention_dropout_rate=attention_dropout_rate,
                                            activation_dropout_rate=activation_dropout_rate,
                                            activation_fn=activation_fn,
                                            layer_norm_first=layer_norm_first) for _ in range(num_encoder_layers)
        ])

        self.layer_norm_first = layer_norm_first
        self.layer_norm = LayerNorm((self.embedding_dim,), epsilon=1e-5).to_float(mstype.float32)

        self.dropout = Dropout(1 - self.dropout_rate)

        self.layer_drop_rate = encoder_layer_drop_rate
        # TODO: Implement layer drop.
        
        self.astp = FAS(in_dim=self.embedding_dim, bottleneck_dim=256)
        self.transpose = ops.Transpose()
        self.spk_norm = spk_norm
        if spk_norm:
            self.spk_norm_layer = LayerNorm((self.embedding_dim,), epsilon=1e-5).to_float(mstype.float32)

    @staticmethod
    def index_put(tensor, indices, value):
        # TODO: Extraction as a public method
        for _ in range(indices.ndim, tensor.ndim):
            indices = indices.expand_dims(-1)
        if indices.shape[-1] < tensor.shape[-1]:
            indices = indices.expand_as(tensor)
        tensor = ops.mul(tensor, ~indices) + ops.mul(value, indices)

        return tensor

    def extract_feature(self, x, padding_mask=None, output_layer=None, mask=None, mask_emb=None):
        """Extract feature."""
        if padding_mask is not None:
            x = self.index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.swapaxes(1, 2))
        x_conv = x_conv.swapaxes(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.dropout(x)

        # BxTxC -> TxBxC
        x = x.swapaxes(0, 1)
        
        # ss
        inter_out = []
        
        # TODO: Implement layer drop
        for i, layer in enumerate(self.layers):
            x = layer(x, self_attn_padding_mask=padding_mask)

            if i in self.encoder_inter_loss_layer:
                if i == self.encoder_inter_loss_layer[0]:
                    spk_emb = self.astp(self.transpose(x, (1, 2, 0)), padding_mask=padding_mask)
                    spk_emb = self.transpose(spk_emb, (2, 0, 1))
                    inter_out.append(spk_emb.swapaxes(0, 1))
                    x = x - spk_emb
                    if self.spk_norm:
                        x = self.spk_norm_layer(x)
                        
        # TxBxC -> BxTxC
        x = x.swapaxes(0, 1)
        return x, inter_out

    def construct(self, x, padding_mask=None, mask=None, output_layer=None, mask_emb=None):
        x, inter_out = self.extract_feature(x, padding_mask, output_layer, mask=mask, mask_emb=mask_emb)

        return x, inter_out
