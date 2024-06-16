import math
from typing import Optional, Tuple
import numpy as np

import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Constant, XavierUniform, initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import constexpr
from .dropout import Dropout
from .dense import Dense


class MultiHeadedAttention(nn.Cell):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, compute_type=mstype.float32):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.neg_inf = Tensor([-10000.0], dtype=compute_type)
        self.scores_mul = Tensor([1.0 / math.sqrt(float(self.d_k))], dtype=compute_type)

        self.linear_q = Dense(n_feat, n_feat).to_float(compute_type)
        self.linear_k = Dense(n_feat, n_feat).to_float(compute_type)
        self.linear_v = Dense(n_feat, n_feat).to_float(compute_type)
        self.linear_out = Dense(n_feat, n_feat).to_float(compute_type)
        self.dropout = Dropout(1 - dropout_rate)
        self.softmax = nn.Softmax()

        self.expand_dims = ops.ExpandDims()
        self.equal = ops.Equal()
        self.matmul = ops.BatchMatMul()
        self.cast = ops.Cast()
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.get_dtype = ops.DType()

    def forward_qkv(self,
                    query: mindspore.Tensor,
                    key: mindspore.Tensor,
                    value: mindspore.Tensor) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """Transform query, key and value.

        Args:
            query (ms.Tensor): Query tensor (#batch, time1, size).
            key (ms.Tensor): Key tensor (#batch, time2, size).
            value (ms.Tensor): Value tensor (#batch, time2, size).

        Returns:
            ms.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            ms.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            ms.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).
        """
        n_batch = query.shape[0]
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(0, 2, 1, 3)  # (batch, head, time1, d_k)
        k = k.transpose(0, 2, 1, 3)  # (batch, head, time2, d_k)
        v = v.transpose(0, 2, 1, 3)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self,
                          value: mindspore.Tensor,
                          scores: mindspore.Tensor,
                          mask: Optional[mindspore.Tensor]) -> mindspore.Tensor:
        """Compute attention context vector.

        Args:
            value (ms.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (ms.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (ms.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            ms.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        n_batch = value.shape[0]

        if mask is not None:
            attn_mask = self.expand_dims(mask, 1)
            attn_mask = self.cast(self.equal(attn_mask, 0), self.get_dtype(scores))
            if len(attn_mask.shape) == 3:
                attn_mask = self.expand_dims(attn_mask, 1)
            attn_mask = self.mul(attn_mask, self.neg_inf)
            scores = self.add(attn_mask, scores)
            attn = self.softmax(scores)  # (batch, head, time1, time2)
        else:
            attn = self.softmax(scores)  # (batch, head, time1, time2)
        p_attn = self.dropout(attn)
        x = self.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(0, 2, 1, 3).view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)  # (batch, time1, d_model)

    def construct(self,
                  query: mindspore.Tensor,
                  key: mindspore.Tensor,
                  value: mindspore.Tensor,
                  mask: Optional[mindspore.Tensor],
                  pos_emb: Optional[mindspore.Tensor] = None) -> mindspore.Tensor:  # pylint: disable=W0613
        """Compute scaled dot product attention.

        Args:
            query (ms.Tensor): Query tensor (#batch, time1, size).
            key (ms.Tensor): Key tensor (#batch, time2, size).
            value (ms.Tensor): Value tensor (#batch, time2, size).
            mask (ms.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.
            pos_emb (ms.Tensor): Positional embedding tensor
                (#batch, time2, size).
        Returns:
            ms.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = self.matmul(q * self.scores_mul, k.transpose(0, 1, 3, 2) * self.scores_mul)

        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.

    Paper: https://arxiv.org/abs/1901.02860/
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate, compute_type):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, compute_type)
        # linear transformation for positional embeddings
        self.linear_pos = Dense(n_feat, n_feat, has_bias=False).to_float(compute_type)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = Parameter(initializer(XavierUniform(), [self.h, self.d_k], mstype.float32))
        self.pos_bias_v = Parameter(initializer(XavierUniform(), [self.h, self.d_k], mstype.float32))
        self.tile = ops.Tile()
        self.norm_factor = Tensor(1.0 / math.sqrt(self.d_k), mindspore.float16)

    def construct(self,
                  query: mindspore.Tensor,
                  key: mindspore.Tensor,
                  value: mindspore.Tensor,
                  mask: Optional[mindspore.Tensor],
                  pos_emb: Optional[mindspore.Tensor] = None):
        """Compute 'Scaled Dot Product Attention' with rel.

        positional encoding.
        Args:
            query (ms.Tensor): Query tensor (#batch, time1, size).
            key (ms.Tensor): Key tensor (#batch, time2, size).
            value (ms.Tensor): Value tensor (#batch, time2, size).
            mask (ms.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            pos_emb (ms.Tensor): Positional embedding tensor
                (#batch, time2, size).
        Returns:
            ms.Tensor: Output tensor (#batch, time1, d_model).
        """
        n_batch = query.shape[0]
        n_batch_pos = pos_emb.shape[0]

        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(0, 2, 1, 3)  # (batch, time1, head, d_k)

        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(0, 2, 1, 3)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.cast(self.pos_bias_u, self.get_dtype(q))).transpose(0, 2, 1, 3)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.cast(self.pos_bias_v, self.get_dtype(q))).transpose(0, 2, 1, 3)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = self.matmul(q_with_bias_u, k.transpose(0, 1, 3, 2))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        p = self.tile(p, (n_batch, 1, 1, 1))
        matrix_bd = self.matmul(q_with_bias_v, p.transpose(0, 1, 3, 2))
        # Remove relative shift of matrix_bd since it is useless in speech recognition,
        # and it requires special attention for streaming.
        scores = matrix_ac + matrix_bd
        scores = self.mul(scores, self.scores_mul)

        return self.forward_attention(v, scores, mask)


@constexpr
def create_neg_inf(shape, dtype):
    if dtype == mstype.float32:
        return Tensor(np.full(shape, -3.4028235e+38, dtype=np.float32))
    if dtype == mstype.float16:
        return Tensor(np.full(shape, -65504, dtype=np.float16))
    raise TypeError('dtype must be float16 or float32.')


class MultiheadAttention(nn.Cell):
    """MultiheadAttention.

    Args:
        embed_dim (int): dimensions of attention blocks
        num_heads (int): number of heads of attention blocks
        k_dim (int | optional): if be configured, k projection will use this as dimension.
        v_dim (int | optional): if be configured, v projection will use this as dimension.
        dropout_rate (float): drop rate of attention result. default: 0.0
        has_bias (bool): whether has bias parameters of kqv projection. default: True
    """

    def __init__(self, embed_dim, num_heads, k_dim=None, v_dim=None, dropout_rate=0.0, has_bias=True):
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.k_dim = k_dim if k_dim is not None else embed_dim
        self.v_dim = v_dim if v_dim is not None else embed_dim
        assert self.embed_dim == self.k_dim == self.v_dim

        self.num_heads = num_heads
        self.dropout_module = Dropout(1 - dropout_rate)

        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim), 'embed_dim must be divisiable by num_heads'
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Dense(self.k_dim, embed_dim, has_bias=has_bias).to_float(mindspore.float16)
        self.v_proj = nn.Dense(self.v_dim, embed_dim, has_bias=has_bias).to_float(mindspore.float16)
        self.q_proj = nn.Dense(embed_dim, embed_dim, has_bias=has_bias).to_float(mindspore.float16)

        self.out_proj = nn.Dense(embed_dim, embed_dim, has_bias=has_bias).to_float(mindspore.float16)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        self.k_proj.weight.set_data(
            initializer(XavierUniform(gain=1 / math.sqrt(2)), self.k_proj.weight.shape, self.k_proj.weight.dtype))
        self.v_proj.weight.set_data(
            initializer(XavierUniform(gain=1 / math.sqrt(2)), self.v_proj.weight.shape, self.v_proj.weight.dtype))
        self.q_proj.weight.set_data(
            initializer(XavierUniform(gain=1 / math.sqrt(2)), self.q_proj.weight.shape, self.q_proj.weight.dtype))
        self.out_proj.weight.set_data(initializer(XavierUniform(), self.out_proj.weight.shape,
                                                  self.q_proj.weight.dtype))
        if self.out_proj.has_bias:
            self.out_proj.bias.set_data(initializer(Constant(0.0), self.out_proj.bias.shape, self.out_proj.bias.dtype))

    def construct(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """Construct."""
        # attn_mask: ms.bool_
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).swapaxes(0, 1)
        k = k.reshape(-1, bsz * self.num_heads, self.head_dim).swapaxes(0, 1)
        v = v.reshape(-1, bsz * self.num_heads, self.head_dim).swapaxes(0, 1)

        _, src_len, _ = k.shape

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.expand_dims(0)

        if key_padding_mask is not None:
            key_padding_mask = ops.BroadcastTo(
                (-1, self.num_heads, -1, -1))(key_padding_mask.view(bsz, 1, 1,
                                                                    src_len)).reshape(bsz * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = ops.logical_or(attn_mask, key_padding_mask)

        q = q * self.scaling

        attn = ops.BatchMatMul()(q, k.swapaxes(-2, -1))

        if attn_mask is not None:
            attn_type = attn.dtype
            attn = ops.cast(attn, mstype.float32)
            attn = ops.select(attn_mask.expand_as(attn), create_neg_inf(attn.shape, mstype.float32), attn)
            attn = ops.cast(ops.Softmax(-1)(attn), attn_type)
        else:
            attn = ops.cast(ops.Softmax(-1)(ops.cast(attn, mindspore.float32)), attn.dtype)
        attn = self.dropout_module(attn)

        attn_output = ops.BatchMatMul()(attn, v)
        attn_output = attn_output.swapaxes(0, 1).reshape(tgt_len * bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.shape[1])

        return attn_output
