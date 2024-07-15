import pyworld as pw
import numpy as np

def extract_pitch(ori_audio):
    # print(save_path, ori_audio.shape)
    frame_period = 320/16000*1000
    f0, timeaxis = pw.dio(ori_audio.astype('float64'), 16000, frame_period=frame_period)
    nonzeros_indices = np.nonzero(f0)
    pitch = f0.copy()
    pitch[nonzeros_indices] = np.log(f0[nonzeros_indices])
    mean, std = np.mean(pitch[nonzeros_indices]), np.std(pitch[nonzeros_indices])
    pitch[nonzeros_indices] = (pitch[nonzeros_indices] - mean) / (std + 1e-8)
    return np.expand_dims(pitch, 0)

def get_feat_extract_output_lengths(input_length, kernel_size, stride):
    """get seqs length after cnns module downsampling."""
    len_ds = input_length
    for i in range(len(kernel_size)):
        len_ds = (len_ds - kernel_size[i]) // stride[i] + 1
    return len_ds

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