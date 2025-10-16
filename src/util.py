import torch
import torch.nn as nn
from config import small_config

def truncated_normal_(tensor, mean=0.0, std=small_config['init_range']):
    size = tensor.size()
    tmp = tensor.new_empty(size + (10,)).normal_(mean=mean, std=std)
    valid = (tmp < mean + 2 * std) & (tmp > mean - 2 * std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    return tensor

def lrate(step_num, d_model, warmup_steps):
    if step_num == 0:
        step_num = 1
    with torch.no_grad():
        step1 = torch.pow(torch.tensor(d_model),-0.5)
        step2 = torch.min(torch.tensor((torch.pow(torch.tensor(step_num),-0.5), step_num*torch.pow(torch.tensor(warmup_steps),-1.5))))
        learning_rate = step1*step2
    return learning_rate