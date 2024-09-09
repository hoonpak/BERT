import sys
sys.path.append("../src")
import argparse

from config import small_config, tiny_config
from tokenizers import Tokenizer
from data import CustomDataset
from model import PretrainingBERT
from util import lrate

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter


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

tokenizer_file_path = "../dataset/BERT_Tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_file_path)
vocab_size = tokenizer.get_vocab_size()
assert vocab_size == config_dict['vocab_size']

pretraining_dataset_file_path = "pretraining_ttest.json"
pretraining_dataset = CustomDataset(pretraining_dataset_file_path, 1, 0, config_dict['pad_idx'])
batch_size = 64
train_dataloader = DataLoader(pretraining_dataset, batch_size)

model = PretrainingBERT(config_dict)
nsp_loss_function = torch.nn.CrossEntropyLoss()
mlm_loss_function = torch.nn.CrossEntropyLoss(ignore_index=config_dict['pad_idx'])
optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9,0.999), weight_decay=0.01)
lr_update = LambdaLR(optimizer=optimizer, lr_lambda=lambda step: lrate(step, config_dict['dim_model'], 10000))

