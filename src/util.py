import torch
import torch.nn as nn
from config import config

def truncated_normal_(tensor, mean=0.0, std=config['init_range']):
    size = tensor.size()
    tmp = tensor.new_empty(size + (10,)).normal_(mean=mean, std=std)
    valid = (tmp < mean + 2 * std) & (tmp > mean - 2 * std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    return tensor