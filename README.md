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
- [x] Codes of [ProgRE](https://github.com/wangtianrui/ProgRE/blob/master/models/progre.py) and [HuBERT](https://github.com/wangtianrui/ProgRE/blob/master/models/hubert.py) model under MindSpore Framework
- [x] [Pretrained checkpoint of Base and Large version ProgRE under MindSpore Framework](https://drive.google.com/drive/folders/1nLsGpXYBsc-kwHKWolDSWISNw3ji4CFY?usp=sharing)
- [x] [Error analysis of migrating MindSpore framework models to Pytorch framework](https://github.com/wangtianrui/ProgRE/tree/master/supplementary_results)
- [x] [Usage of the pre-trained model in the MindSpore/PyTorch framework](https://github.com/wangtianrui/ProgRE?tab=readme-ov-file#usage-under-mindspore)
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
ms_param = load_checkpoint(r'./pretrained_model/base_progre.ckpt')
print(load_param_into_net(model, ms_param))
model.acc_net.set_train(False)
model.acc_net.set_grad(False)
model.acc_net.mask = False

kernel_size = [int(i) for i in config["extractor_conf"]['kernel_size_list']]
stride = [int(i) for i in config["extractor_conf"]['stride_list']]

wav = lib.load(r"./supplementary_results/LJ001-0068.wav", sr=16000)[0]

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

* First, it is necessary to convert the model checkpoint from the MindSpore framework to a PyTorch checkpoint.
```shell
# Please modify the path in the python file before running
python migrating/chkp_conversion.py
```

* Load converted checkpoint and inference

```python
import yaml
from models.pytorch.progre import ProgRE, ProgREConfig
import torch
from copy import deepcopy
import librosa as lib
from models.config import get_config
import pyworld as pw
from models.utils import extract_pitch, make_pad_mask, get_feat_extract_output_lengths
import numpy as np

config_ms = get_config("./config/progre.yaml")

with open("./config/progre_pt.yaml", 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
print(config)
config = ProgREConfig(**config)
model = ProgRE.build_model(config)
ckpt = torch.load(r"pretrained_model/TORCHModel_base_progre.ckpt", map_location="cpu")["model"]
pretrained_dict = {}
model_state = model.state_dict()
torch_names = deepcopy(list(model_state.keys()))
for k, v in ckpt.items():
    if k.find("fbank_encoder") != -1 and k.find("fc") != -1:
        k = k.replace("fc1", "norm1")
        k = k.replace("fc2", "norm2")
    if k.find("spk_astp.out.dense") != -1:
        k = k.replace("dense.", "")
    if k.find("sub_spk_norm") != -1:
        k = k.replace("beta", "bias")
    if k in model_state and k.find("label_embs_concat") == -1 and k.find("final_proj") == -1:
        if v.shape != model_state[k].shape:
            if len(v.shape) != len(model_state[k].shape):
                print("%s convert shape from %s to %s"%(k, str(v.shape), str(model_state[k].shape))) 
                v = v.reshape(model_state[k].shape)
            else:
                print("transpose shape from %s to %s"%(str(v.shape), str(v.T.shape))) 
                v = v.T
        pretrained_dict[k] = v
        torch_names.remove(k)
model_state.update(pretrained_dict)
model.load_state_dict(model_state)
model.eval()
kernel_size = [int(i) for i in config_ms["extractor_conf"]['kernel_size_list']]
stride = [int(i) for i in config_ms["extractor_conf"]['stride_list']]
wav = lib.load(r"./supplementary_results/LJ001-0068.wav", sr=16000)[0]
wav = np.pad(wav, [0, 320-len(wav)%320])
origin_len = len(wav)
inp_len = len(wav)
feature_lens = np.array([get_feat_extract_output_lengths(origin_len, kernel_size, stride), ])
feature_paded_len = get_feat_extract_output_lengths(inp_len, kernel_size, stride)
pitch = extract_pitch(wav)[:, :feature_paded_len]
feature, hidden_layers = model(
    source=torch.tensor(wav[None, :], dtype=torch.float),
    padding_mask=torch.BoolTensor(make_pad_mask(feature_lens, max_len=feature_paded_len)),
    pitch=torch.tensor(pitch, dtype=torch.float),
    features_only=True,
)
print(feature.shape)
```