from typing import Optional

import mindspore
import numpy as np
from mindspore import dtype as mstype
from mindspore import nn, ops
from mindspore.common.initializer import Uniform, initializer
from mindspore.common.tensor import Tensor
from mindspore.nn import DynamicLossScaleUpdateCell
from mindspore.nn.wrap import TrainOneStepWithLossScaleCell
from mindspore.ops import composite as C
from mindspore.ops import constexpr
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.initializer import HeUniform, Uniform, _calculate_correct_fan, initializer
from mindspore.nn.cell import Cell
import math

from .feature_extractor import ConvFeatureExtractionModel
from .layernorm import LayerNorm
from .encoder import Wav2VecTransformerEncoder
from .dropout import Dropout


class Dense(Cell):
    """A self-defined dense layer.

    Args:
        in_channel (int): The number of channels in the input space.
        out_channel (int): The number of channels in the output space.
        has_bias (bool): The trainable bias_init parameter.
        activation (nn.Cell): activate function applied to the output of the fully connected layer.
        negative_slope (int, float, bool): The negative slope of the rectifier used after this layer.
        mode (str): Either "fan_in" or "fan_out".
        nonlinerity (str): The non-linear function, use only with "relu" or "leaky_relu"
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 has_bias=True,
                 activation=None,
                 negative_slope=math.sqrt(5),
                 mode='fan_in',
                 nonlinearity='leaky_relu',
                 type=mindspore.float32):
        super(Dense, self).__init__()
        kaiming_uniform_0 = initializer(HeUniform(negative_slope=negative_slope, mode=mode, nonlinearity=nonlinearity),
                                        (out_channel, in_channel))
        bias_init_0 = 'zeros'
        if has_bias:
            fan_in = _calculate_correct_fan((out_channel, in_channel), mode=mode)
            scale = 1 / math.sqrt(fan_in)
            bias_init_0 = initializer(Uniform(scale), [out_channel], type)
        self.dense = nn.Dense(in_channel,
                              out_channel,
                              weight_init=kaiming_uniform_0,
                              bias_init=bias_init_0,
                              has_bias=True,
                              activation=activation).to_float(type)

    def construct(self, x):
        out = self.dense(x)
        return out

@constexpr
def create_neg_inf(shape, dtype):
    if dtype == mstype.float32:
        return Tensor(np.full(shape, -3.4028235e+38, dtype=np.float32))
    if dtype == mstype.float16:
        return Tensor(np.full(shape, -65504, dtype=np.float16))
    raise TypeError('dtype must be float16 or float32.')


class HubertModel(nn.Cell):
    """Definition of Hubert speech pretrained model.

    Args:
        enc_input_dim (int): Dimension of encoder input features.
        enc_output_dim (int): Dimension of encoder output features.
        n_classes (int): Number of kmeans clusters.
        feature_extractor (nn.Cell): Feature extractor instance.
        encoder (nn.Cell): Encoder instance.
        n_classes (int): Number of pesudo label clusters.
        final_dim (int): Output dimension of Hubert model.
        logit_temp (float): scale factor for logits.
        dropout_input (float): Dropout rate for dropping input.
        dropout_features (float): Dropout rate for dropping features.
        pred_masked_weight (float): Loss weight of masked prediction.
        pred_unmasked_weight (float): Loss weight of unmasked prediction.
        feature_pen_weight (float): Loss weight of feature penalty.
        encoder_type (str): encoder type.
        compute_type (int): Whether to use mix precision training.
    """

    def __init__(self,
                 enc_input_dim,
                 enc_output_dim,
                 feature_extractor,
                 encoder,
                 n_classes,
                 final_dim=2048,
                 logit_temp=0.1,
                 dropout_input=0.1,
                 dropout_features=0.1,
                 pred_masked_weight=1.0,
                 pred_unmasked_weight=0.0,
                 feature_pen_weight=10,
                 encoder_type='conformer',
                 compute_type=mindspore.float16,
                 feature_grad_mult=0.1,
                 stride_list=None,
                 label_rate=None,
                 sample_rate=16000,
                 mask_conf=None):
        super(HubertModel, self).__init__()
        # definition of modules
        self.feature_extractor = feature_extractor
        self.layer_norm = LayerNorm((enc_input_dim,), epsilon=1e-5).to_float(mstype.float32)
        self.post_extract_proj = (Dense(enc_input_dim, enc_output_dim).to_float(compute_type)
                                  if enc_input_dim != enc_output_dim else None)
        self.encoder = encoder
        self.dropout_input = Dropout(1.0 - dropout_input)
        self.dropout_feature = Dropout(1.0 - dropout_features)
        self.squeeze = ops.Squeeze(1)
        self.n_classes = n_classes
        self.logit_temp = logit_temp
        self.pred_masked_weight = pred_masked_weight
        self.pred_unmasked_weight = pred_unmasked_weight
        self.feature_pen_weight = feature_pen_weight
        self.encoder_type = encoder_type

    def forward_align(
            self,
            source: Tensor,
            x_len: Tensor,
            output_layer: Optional[int] = None,
            padding_mask=None
    ) -> Tensor:
        # np.save("/Work20/2023/wangtianrui/codes/projects/ProgRE/supplementary_results/ms/ms_conv_inp.npy", source.asnumpy())
        features = self.feature_extractor(source, x_len)
        # np.save("/Work20/2023/wangtianrui/codes/projects/ProgRE/supplementary_results/ms/ms_conv_out.npy", features.asnumpy())
        features = features.transpose(0, 2, 1)
        features = self.layer_norm(features)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        # conv_out = features.copy()
        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        # np.save("/Work20/2023/wangtianrui/codes/projects/ProgRE/supplementary_results/ms/ms_encoder_inp.npy", features.asnumpy())
        features, results = self.encoder(
            x=features,
            padding_mask=padding_mask,
            output_layer=None if output_layer is None else output_layer - 1,
        )
        # results.insert(0, conv_out)
        # np.save("/Work20/2023/wangtianrui/codes/projects/ProgRE/supplementary_results/ms/ms_encoder_out.npy", features.asnumpy())
        return features, results

class HuBERTTraining(nn.Cell):
    def __init__(self,
                 enc_input_dim,
                 enc_output_dim,
                 feature_extractor,
                 encoder,
                 n_classes,
                 final_dim=2048,
                 logit_temp=0.1,
                 dropout_input=0.1,
                 dropout_features=0.1,
                 pred_masked_weight=1.0,
                 pred_unmasked_weight=0.0,
                 feature_pen_weight=10,
                 encoder_type='transformer',
                 compute_type=mindspore.float16,
                 feature_grad_mult=0.1,
                 stride_list=None,
                 label_rate=None,
                 mask_conf=None):
        super(HuBERTTraining, self).__init__()
        self.acc_net = HubertModel(enc_input_dim=enc_input_dim,
                                        enc_output_dim=enc_output_dim,
                                        feature_extractor=feature_extractor,
                                        encoder=encoder,
                                        n_classes=n_classes,
                                        final_dim=final_dim,
                                        logit_temp=logit_temp,
                                        dropout_input=dropout_input,
                                        dropout_features=dropout_features,
                                        pred_masked_weight=pred_masked_weight,
                                        pred_unmasked_weight=pred_unmasked_weight,
                                        feature_pen_weight=feature_pen_weight,
                                        encoder_type=encoder_type,
                                        compute_type=compute_type,
                                        feature_grad_mult=feature_grad_mult,
                                        stride_list=stride_list,
                                        label_rate=label_rate,
                                        mask_conf=mask_conf)

    def construct(self, xs, xs_len, ys, padding_mask, hubert_mask, hubert_unmask):
        loss = self.acc_net(xs, xs_len, ys, padding_mask, hubert_mask, hubert_unmask)
        return loss

def init_hubert_model(config, input_dim):
    """Init a Hubert model."""
    extractor_extra_args = config['extractor_conf']
    hubert_extra_args = config['hubert_conf']
    encoder_extra_args = config['encoder_conf']
    label_rate = extractor_extra_args['label_rate']
    mask_conf = config["collate_conf"]["mask_conf"]

    feature_extractor = ConvFeatureExtractionModel(stride_list=extractor_extra_args['stride_list'],
                                                   kernel_size_list=extractor_extra_args['kernel_size_list'],
                                                   dim=extractor_extra_args['output_dim'],
                                                   dropout=extractor_extra_args['dropout'],
                                                   layer_norm=extractor_extra_args['layer_norm'],
                                                   has_bias=extractor_extra_args['has_bias'],
                                                   compute_type=mindspore.float32)
    encoder = Wav2VecTransformerEncoder(encoder_embed_dim=encoder_extra_args['encoder_embed_dim'],
                                        encoder_ffn_embed_dim=encoder_extra_args['encoder_ffn_embed_dim'],
                                        encoder_num_attention_heads=encoder_extra_args['encoder_attention_heads'],
                                        num_encoder_layers=encoder_extra_args['encoder_layers'],
                                        conv_pos_kernel_size=encoder_extra_args['conv_pos'],
                                        conv_pos_groups=encoder_extra_args['conv_pos_groups'],
                                        conv_pos_use_same_pad=encoder_extra_args['conv_pos_use_same_pad'],
                                        pos_conv_depth=encoder_extra_args['pos_conv_depth'],
                                        dropout_rate=encoder_extra_args['dropout_rate'],
                                        attention_dropout_rate=encoder_extra_args['attention_dropout_rate'],
                                        activation_dropout_rate=encoder_extra_args['activation_dropout_rate'],
                                        activation_fn=encoder_extra_args['activation_fn'],
                                        layer_norm_first=encoder_extra_args['layer_norm_first'],
                                        required_seq_len_multiple=encoder_extra_args['required_seq_len_multiple'],
                                        encoder_layer_drop_rate=encoder_extra_args['encoder_layerdrop'],
                                        compute_type=mindspore.float16)
    model = HuBERTTraining(hubert_extra_args['input_dim'],
                        hubert_extra_args['output_dim'],
                        feature_extractor=feature_extractor,
                        encoder=encoder,
                        n_classes=hubert_extra_args['n_classes'],
                        final_dim=hubert_extra_args['final_dim'],
                        logit_temp=hubert_extra_args['logit_temp'],
                        dropout_input=hubert_extra_args['dropout_input'],
                        dropout_features=hubert_extra_args['dropout_features'],
                        pred_masked_weight=hubert_extra_args['pred_masked_weight'],
                        pred_unmasked_weight=hubert_extra_args['pred_unmasked_weight'],
                        feature_pen_weight=hubert_extra_args['feature_pen_weight'],
                        encoder_type=config['encoder'],
                        compute_type=mindspore.float16,
                        feature_grad_mult=extractor_extra_args['feature_grad_mult'],
                        stride_list=extractor_extra_args['stride_list'],
                        label_rate=label_rate,
                        mask_conf=mask_conf)
    return model

