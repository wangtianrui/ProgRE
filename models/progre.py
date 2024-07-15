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
import mindspore.numpy as ms_np
from .dense import Dense
from .feature_extractor import ConvFeatureExtractionModel
from .layernorm import LayerNorm
from .progre_encoder import ProgRETransformerEncoder
from .dropout import Dropout
from .conv1d import Conv1d

class PitchExtractor(nn.Cell):
    def __init__(self, out_dim, compute_type):
        super().__init__()
        self.compute_type = compute_type
        self.conv1 = nn.SequentialCell([
            nn.Conv1d(
                in_channels=1,
                out_channels=256,
                kernel_size=5,
                has_bias=True,
                pad_mode="same"
            ).to_float(compute_type), 
            nn.BatchNorm1d(256).to_float(mindspore.float32), 
            nn.ReLU()
        ])
        self.conv2 = nn.SequentialCell([
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=5,
                has_bias=True,
                pad_mode="same"
            ).to_float(compute_type), 
            nn.BatchNorm1d(256).to_float(mindspore.float32), 
            nn.ReLU()
        ])
        self.conv3 = nn.SequentialCell([
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=5,
                has_bias=True,
                pad_mode="same"
            ).to_float(compute_type), 
            nn.BatchNorm1d(256).to_float(mindspore.float32), 
            nn.ReLU()
        ])
        self.lstm = nn.GRU(input_size=256, hidden_size=32, 
                            num_layers=1, batch_first=True, bidirectional=True).to_float(compute_type)
        self.expand_dims = ops.ExpandDims()
        self.out = Dense(in_channel=32*2, out_channel=out_dim).to_float(compute_type)
        
    def construct(self, x, padding_mask):
        x = self.expand_dims(x, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x, _ = self.lstm(x.swapaxes(1, 2))
        return self.out(x) * self.expand_dims((1.0-padding_mask.astype(self.compute_type)), -1)
    
class ProgRE(nn.Cell):
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
        super(ProgRE, self).__init__()

        # definition of modules
        self.feature_extractor = feature_extractor
        self.layer_norm = LayerNorm((enc_input_dim,), epsilon=1e-5).to_float(mstype.float32)

        self.post_extract_proj = (Dense(enc_input_dim, enc_output_dim).to_float(compute_type)
                                  if enc_input_dim != enc_output_dim else None)
        self.encoder = encoder
        
        self.pitch_encoder = PitchExtractor(out_dim=enc_output_dim, compute_type=mstype.float32)
        self.sub_norm = LayerNorm((enc_output_dim,), epsilon=1e-5).to_float(mstype.float32)
        
        self.dropout_input = nn.Dropout(1.0 - dropout_input)
        self.dropout_feature = nn.Dropout(1.0 - dropout_features)
        self.squeeze = ops.Squeeze(1)
        self.n_classes = n_classes
        self.logit_temp = logit_temp
        self.encoder_type = encoder_type

    def forward_align(self, source, x_len, padding_mask, pitch, output_layer=None):
        features = self.feature_extractor(source, x_len)
        features = features.transpose(0, 2, 1)  # (B, T, C)
        features = self.layer_norm(features)
        T = features.shape[1]
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        pitch_emb = self.pitch_encoder(pitch, padding_mask=padding_mask)
        features = self.sub_norm(features - pitch_emb)
        h_enc, hidden_layers = self.encoder(features, padding_mask=padding_mask.copy())
        return h_enc, hidden_layers
    
class ProgRETraining(mindspore.nn.Cell):
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
        super(ProgRETraining, self).__init__()
        self.acc_net = ProgRE(enc_input_dim=enc_input_dim,
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
    
    def construct(self, xs, xs_len, ys, padding_mask, wavlm_mask, spk_labels, pitches):
        loss = self.acc_net(xs, xs_len, ys, padding_mask, wavlm_mask, spk_labels, pitches)
        return loss


def init_progre_model(config, input_dim):
    extractor_extra_args = config['extractor_conf']
    progre_extra_args = config['progre_conf']
    encoder_extra_args = config['encoder_conf']
    label_rate = extractor_extra_args['label_rate']
    mask_conf = config["collate_conf"]["mask_conf"]

    feature_extractor = ConvFeatureExtractionModel(stride_list=extractor_extra_args['stride_list'],
                                                   kernel_size_list=extractor_extra_args['kernel_size_list'],
                                                   dim=extractor_extra_args['output_dim'],
                                                   dropout=extractor_extra_args['dropout'],
                                                   layer_norm=extractor_extra_args['layer_norm'],
                                                   has_bias=extractor_extra_args['has_bias'])

    encoder = ProgRETransformerEncoder(encoder_embed_dim=encoder_extra_args['encoder_embed_dim'],
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
                                        encoder_inter_loss_layer=encoder_extra_args['encoder_inter_loss_layer'])
   

    model = ProgRETraining(progre_extra_args['input_dim'],
                        progre_extra_args['output_dim'],
                        feature_extractor=feature_extractor,
                        encoder=encoder,
                        n_classes=progre_extra_args['n_classes'],
                        final_dim=progre_extra_args['final_dim'],
                        logit_temp=progre_extra_args['logit_temp'],
                        dropout_input=progre_extra_args['dropout_input'],
                        dropout_features=progre_extra_args['dropout_features'],
                        pred_masked_weight=progre_extra_args['pred_masked_weight'],
                        pred_unmasked_weight=progre_extra_args['pred_unmasked_weight'],
                        feature_pen_weight=progre_extra_args['feature_pen_weight'],
                        encoder_type=config['encoder'],
                        compute_type=mindspore.float16,
                        feature_grad_mult=extractor_extra_args['feature_grad_mult'],
                        stride_list=extractor_extra_args['stride_list'],
                        label_rate=label_rate,
                        mask_conf=mask_conf)

    return model
