# This folder primarily contains auxiliary materials for our paper

* Analysis of Migration Errors: Analyzes errors encountered when migrating a model (demonstrated with a HuBERT Base model) from the MindSpore framework to the PyTorch framework, using calculations and visualizations.
* Chinese Speech Recognition Performance: Demonstrates the performance of the proposed model on the Chinese speech recognition task.


## Migration Errors
We directly convert the parameters of the HuBERT model trained based on the MindSpore framework into a format readable by PyTorch. Then, using the same parameters, we extract features from the same audio using HuBERT built with MindSpore and HuBERT built with PyTorch, respectively. We saved the input to the encoder and the output of each layer of the encoder, totaling 13 layers of representations for comparison.

First, we visualize the representations of the 13 layers and the error. The first column is the result from MindSpore, the second column is the result from PyTorch, and the third column is the error with the formula `diff = abs(ms - pt)`.

![](./test.png)

Next, we show the relative error, calculated as `np.mean(abs(ms - pt)) / np.mean(abs(ms))`, with the result being as follow. 
```txt
0 0.0005469106
1 0.0008232473
2 0.0015528084
3 0.0020986304
4 0.00258094
5 0.0024200461
6 0.0024519553
7 0.0021557575
8 0.0020926956
9 0.0019156568
10 0.002052982
11 0.0021666065
12 0.002720383
```
It can be seen that the migration introduces approximately a 0.25% relative error.
This causes a slight performance degradation when the model pre-trained based on MindSpore is used in a fine-tuning framework based on PyTorch.



## Chinese ASR
In our paper, we mentioned the capability of processing Chinese speech. To further verify this, we evaluated the Chinese speech recognition performance on the AISHELL1 dataset, results are shown as follow,

| Method | Pre-training Data | CER on AISHELL1 |
|:----------:|:----------:| :----------:|
| HuBERT Large Fairseq | LibriLight 60k | 3.9 |
| HuBERT Large MindSpore | MLS 40k + Wenetspeech 10k + Chinese data from Internet 30k | 3.6 |
| ProgRE Large MindSpore | MLS 40k + Wenetspeech 10k + Chinese data from Internet 30k | 3.2 |

We performed fine-tuning on the AISHELL1 dataset based on the Espnet framework. The detailed configuration for the fine-tuning is as follows:
```yaml
num_workers: 8
batch_type: numel
batch_bins: 8000000
accum_grad: 1
max_epoch: 50
patience: 20
num_att_plot: 0
init: none
normalize: none
val_scheduler_criterion:
    - valid
    - acc
best_model_criterion:
-   - valid
    - acc
    - max
keep_nbest_models: 10
unused_parameters: true
freeze_param: [
"frontend.upstream"
]

frontend: s3prl
frontend_conf:
    frontend_conf:
        upstream: progre_large  
        path_or_url: /Work20/2023/wangtianrui/model_temp/progre_large/TORCHModel_converted_from_ms_s3prl.pt
        # upstream: hubert_local  
        # path_or_url: /Work20/2023/wangtianrui/model_temp/hubert/hubert_large_ll60k_s3prl.pt
        # upstream: hubert_local  
        # path_or_url: /Work20/2023/wangtianrui/model_temp/hubertms_large/TORCHModel_converted_from_ms_s3prl.pt
    multilayer_feature: true

preencoder: linear
preencoder_conf:
    input_size: 1024  
    output_size: 80

encoder: conformer
encoder_conf:
    output_size: 256  
    attention_heads: 4
    linear_units: 2048 
    num_blocks: 12     
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    attention_dropout_rate: 0.0
    input_layer: conv2d2 
    normalize_before: true
    rel_pos_type: latest
    pos_enc_layer_type: rel_pos
    selfattention_layer_type: rel_selfattn
    activation_type: swish
    cnn_module_kernel: 15

decoder: transformer
decoder_conf:
    attention_heads: 4
    linear_units: 2048
    num_blocks: 6
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    self_attention_dropout_rate: 0.0
    src_attention_dropout_rate: 0.0

model_conf:
    ctc_weight: 0.3
    lsm_weight: 0.1    
    length_normalized_loss: false

optim: adam
optim_conf:
   lr: 0.0005
scheduler: warmuplr
scheduler_conf:
   warmup_steps: 30000

specaug: specaug
specaug_conf:
    apply_time_warp: true
    time_warp_window: 5
    time_warp_mode: bicubic
    apply_freq_mask: true
    freq_mask_width_range:
    - 0
    - 30
    num_freq_mask: 2
    apply_time_mask: true
    time_mask_width_range:
    - 0
    - 40
    num_time_mask: 2

```
