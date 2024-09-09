import sys
sys.path.append("../src")
import argparse

from config import small_config, tiny_config
from data import CustomDataset

from tokenizers import Tokenizer

import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="tiny", choices=["tiny", "small"])
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--version")
option = parser.parse_args()

config_dict = tiny_config
if option.model == "small":
    config_dict = small_config

current_name = option.model + "_" + option.version

step_batch_size = 256
num_total_steps = 1000000

