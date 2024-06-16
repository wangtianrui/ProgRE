from models.hubert import init_hubert_model
from config.config import get_config
import librosa as lib
import numpy as np
import mindspore
from mindspore import load_checkpoint, load_param_into_net, nn, save_checkpoint, Tensor, ops
import s3prl.hub as hub
import torch
import matplotlib.pyplot as plt
cast = ops.Cast()

def make_pad_mask(lengths, max_len: int = 0):
    """Make mask containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (List[int]): Batch of lengths (B,).
    Returns:
        np.ndarray: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    # lengths:(B,)
    batch_size = int(len(lengths))
    max_len = max_len if max_len > 0 else max(lengths)
    # np.arange(0, max_len): [0,1,2,...,max_len-1]
    # seq_range (1, max_len)
    seq_range = np.expand_dims(np.arange(0, max_len), 0)
    # seq_range_expand (B,max_len)
    seq_range_expand = np.tile(seq_range, (batch_size, 1))
    # (B,1)
    seq_length_expand = np.expand_dims(lengths, -1)
    # (B,max_len)
    mask = seq_range_expand >= seq_length_expand
    return mask


def get_feat_extract_output_lengths(input_length, kernel_size, stride):
    """get seqs length after cnns module downsampling."""
    len_ds = input_length
    for i in range(len(kernel_size)):
        len_ds = (len_ds - kernel_size[i]) // stride[i] + 1
    return len_ds

def load_models_and_extract_features(save_home):
    ms_ckpt = r"/Work20/2023/wangtianrui/model_temp/align_test/hubert_demo_96epochLS.ckpt"
    pt_ckpt = r"/Work20/2023/wangtianrui/model_temp/align_test/hubert_demo_96epochLS_converted_to_s3prl.pt"
    
    ms_cfg = get_config(r"config/hubert.yaml")
    ms_model = init_hubert_model(ms_cfg, input_dim=ms_cfg['hubert_conf']['input_dim'])
    ms_param = load_checkpoint(ms_ckpt)
    print(load_param_into_net(ms_model, ms_param))
    ms_model.acc_net.set_train(False)
    ms_model.acc_net.set_grad(False)
    # print(ms_model)

    kernel_size = [int(i) for i in ms_cfg["extractor_conf"]['kernel_size_list']]
    stride = [int(i) for i in ms_cfg["extractor_conf"]['stride_list']]

    wav = lib.load(r"supplementary_results/LJ001-0068.wav", sr=16000)[0]
    wav = np.pad(wav, [0, 320-len(wav)%320])
    origin_len = len(wav)
    inp_len = len(wav)
    feature_lens = np.array([get_feat_extract_output_lengths(origin_len, kernel_size, stride), ])
    feature_paded_len = get_feat_extract_output_lengths(inp_len, kernel_size, stride)

    feature, results = ms_model.acc_net.forward_align(
        source=cast(Tensor(wav[None, :]), mindspore.float32),
        x_len=Tensor(feature_lens),
        padding_mask=Tensor(make_pad_mask(feature_lens, max_len=feature_paded_len)).bool(),
    )
    for i in range(len(results)):
        np.save("%s/ms/ms_out%d.npy"%(save_home, i), results[i].asnumpy())
    print(feature.shape, len(results), results[0].shape)
    
    ps_model = getattr(hub, "hubert_local")(pt_ckpt).eval()
    pt_results = ps_model(torch.Tensor([wav]))["hidden_states"]
    for i in range(len(pt_results)):
        np.save("%s/torch/torch_out%d.npy"%(save_home, i), pt_results[i].detach().numpy())
        
if __name__ == "__main__":
    save_home = r"supplementary_results"
    load_models_and_extract_features(save_home=save_home)
    
    # 创建一个子图布局
    fig, axes = plt.subplots(13, 3, figsize=(18, 65)) 
    axes = axes.flatten()
    
    for i in range(13):
        ms = np.load("%s/ms/ms_out%d.npy"%(save_home, i))[0].T
        pt = np.load("%s/torch/torch_out%d.npy"%(save_home, i))[0].T
        
        ax = axes.flat[i*3]
        ax.imshow(ms, cmap='viridis')
        ax.set_title('ms out %d'%i, fontsize=10) 
        ax = axes.flat[i*3+1]
        ax.imshow(pt, cmap='viridis')
        ax.set_title('pt out %d'%i, fontsize=10) 
        ax = axes.flat[i*3+2]
        ax.imshow(abs(ms-pt), cmap='viridis')
        ax.set_title('diff (|ms-pt|) %d'%i, fontsize=10) 
        print(i, np.mean(abs(ms-pt))/np.mean(abs(ms)))
    
    """
    print(i, np.mean(abs(ms-pt))/np.mean(abs(ms))) results:
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
    """
    
    # 调整子图之间的间距
    plt.tight_layout(pad=2.0)

    # 显示图像
    plt.savefig(r"supplementary_results/test.png", dpi=100, bbox_inches='tight')