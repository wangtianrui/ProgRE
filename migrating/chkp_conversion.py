import numpy as np
import torch
from mindspore import dtype as mstype
from mindspore import Parameter, Tensor
from mindspore import load_checkpoint
from copy import copy, deepcopy

def numpy2mstensor(numpy_data, ms_tensor):
    numpy_data = deepcopy(numpy_data)
    ms_tensor = deepcopy(ms_tensor)
    tgt_shape = ms_tensor.shape
    ori_shape = numpy_data.shape
    if tgt_shape != ori_shape:
        if len(tgt_shape) != len(ori_shape):
            numpy_data = numpy_data.reshape(tgt_shape)
            print("convert shape from %s to %s"%(str(ori_shape), str(tgt_shape))) 
        else:
            numpy_data = numpy_data.T
            print("transpose shape from %s to %s"%(str(ori_shape), str(tgt_shape)))
            
    return Parameter(Tensor(numpy_data, ms_tensor.dtype), name=ms_tensor.name, requires_grad=ms_tensor.requires_grad)

def numpy2torchtensor(tensor: np.ndarray, torch_tensor):
    tensor = deepcopy(tensor)
    if torch_tensor is not None:
        torch_tensor = deepcopy(torch_tensor)
        tgt_shape = torch_tensor.shape
        ori_shape = tensor.shape
        tensor = tensor.astype(np.float32)
        if tgt_shape != ori_shape:
            if len(tgt_shape) != len(ori_shape):
                tensor = tensor.reshape(tgt_shape)
                print("convert shape from %s to %s"%(str(ori_shape), str(tgt_shape)))
            else:
                tensor = tensor.T
                print("transpose shape from %s to %s"%(str(ori_shape), str(tgt_shape)))
    return torch.Tensor(tensor)

def get_align_info():
    mf_txt = r"./migrating/ms.txt"
    torch_txt = r"./migrating/torch.txt"
    mf2torch_dict = {}
    torch2mf_dict = {}
    with open(mf_txt, "r") as mf_txt:
        with open(torch_txt, "r") as torch_txt:
            mf_lines = mf_txt.readlines()
            torch_lines = torch_txt.readlines()
            for idx in range(len(mf_lines)):
                mf_name = mf_lines[idx].strip()
                torch_name = torch_lines[idx].strip()
                mf2torch_dict[mf_name] = torch_name
                torch2mf_dict[torch_name] = mf_name
    return mf2torch_dict, torch2mf_dict

if __name__ == "__main__":
    ms_param = load_checkpoint(r"./pretrained_model/base_progre.ckpt")
    print(ms_param.keys())
    mf2torch_param = {"model":{}}
    mf2torch_dict, torch2mf_dict = get_align_info()
    keys = list(deepcopy(list(ms_param.keys())))
    
    for mf_name in mf2torch_dict.keys():
        torch_name = mf2torch_dict[mf_name]
        if torch_name in mf2torch_param["model"].keys():
            torch_tensor = mf2torch_param["model"][torch_name]
        else:
            print("%s is new param from ms, is converted into %s" % (mf_name, torch_name))
            torch_tensor = None
        mf2torch_param["model"][torch_name] = numpy2torchtensor(ms_param[mf_name].asnumpy(), torch_tensor)
        keys.remove(mf_name)

    print("not convert keys: " + str(keys))
    torch.save(mf2torch_param, './pretrained_model/TORCHModel_base_progre.ckpt')
