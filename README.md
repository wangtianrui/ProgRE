<div align="center">
    <h1>
    ProgRE
    </h1>
    <p>
    This is the official implement of Progressive Residual Extraction based Pre-training for Speech Representation Learning  <br>
    </p>
    <!-- <p>
    <img src="docs/logo.png" alt="emobox Logo" style="width: 580px; height: 200px;">
    </p> -->
    <p>
    </p>
</div>

## Guides

**Prog**ressive **R**esidual **E**xtraction based Pre-training (ProgRE) is a speech self-supervised learning model that can progressively extract pitch variation, speaker information, and content information from speech in a residual manner. ProgRE achieves joint performance improvements on various tasks, such as speaker identification, speech recognition, emotion recognition, speech enhancement, and voice conversion.

## Requirement

```shell
conda create -n progre python=3.8
conda activate progre
pip install -r requirements.txt
```

## Todo List
- [x] ProgRE and HuBERT model under MindSpore Framework
- [x] Pretrained checkpoint of Base version ProgRE and HuBERT under MindSpore Framework
- [x] [Error analysis of migrating MindSpore framework models to Pytorch framework](https://github.com/wangtianrui/ProgRE/tree/master/supplementary_results)
- [x] How to use the pre-trained model in the PyTorch framework
- [ ] Training codes under MindSpore Framework
- [ ] Pretrained checkpoint of Large version ProgRE and HuBERT under MindSpore Framework (84,500 hours)
- [ ] Release 84,500 hours English-Chinese pre-training dataset

## Usage under MindSpore
Mindspore-GPU is highly dependent on Ascend GPUs, so we recommend installing the CPU version for testing.

```python
import librosa as lib
from models.config import get_config
import pyworld as pw
from models.utils import extract_pitch, make_pad_mask, get_feat_extract_output_lengths
from models.progre import init_progre_model
from mindspore import load_checkpoint, load_param_into_net, Tensor, ops
import numpy as np
import mindspore

config = get_config("./config/progre.yaml")
print(config)
model = init_progre_model(config, input_dim=config['progre_conf']['input_dim'])
print(model)
ms_param = load_checkpoint(r'E:\data_models\models\speech_split\ss_pitch_all\CKP-37_8240.ckpt')
print(load_param_into_net(model, ms_param))
model.acc_net.set_train(False)
model.acc_net.set_grad(False)
model.acc_net.mask = False

kernel_size = [int(i) for i in config["extractor_conf"]['kernel_size_list']]
stride = [int(i) for i in config["extractor_conf"]['stride_list']]

wav = lib.load(r"E:\codes\ProgRE\supplementary_results\LJ001-0068.wav", sr=16000)[0]

wav = np.pad(wav, [0, 320-len(wav)%320])
origin_len = len(wav)
inp_len = len(wav)
cast = ops.Cast()
feature_lens = np.array([get_feat_extract_output_lengths(origin_len, kernel_size, stride), ])
feature_paded_len = get_feat_extract_output_lengths(inp_len, kernel_size, stride)
pitch = extract_pitch(wav)[:, :feature_paded_len]
feature = model.acc_net.forward_align(
    source=cast(Tensor(wav[None, :]), mindspore.float32),
    x_len=Tensor(feature_lens),
    padding_mask=Tensor(make_pad_mask(feature_lens, max_len=feature_paded_len)).bool(),
    pitch=cast(Tensor(pitch), mindspore.float32),
    output_layer=None,
)
print(feature.shape)
```


## Usage under PyTorch via migrating