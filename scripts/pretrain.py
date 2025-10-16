import sys
sys.path.append("../src")
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

from config import *
from transformers import BertTokenizerFast
from data import PretrainingCustomDataset
from model import PretrainingBERT
from util import lrate

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="tiny", choices=["tiny", "small"])
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--batchsize", default=256, type=int)
parser.add_argument("--version")
option = parser.parse_args()

device = option.device

config_dict = tiny_config
if option.model == "small":
    config_dict = small_config

current_name = option.model + "_" + option.version

print(f"{current_name} READY !!!")

step_batch_size = 256
num_total_steps = 1000000
warmup_steps = 10000
learning_rate = 1e-4

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', use_fast=True)
pad_idx = tokenizer.convert_tokens_to_ids("[PAD]")

vocab_size = tokenizer.vocab_size
print("Tokenizer READY !!!")
assert (vocab_size) == config_dict['vocab_size']

batch_size = option.batchsize

model = PretrainingBERT(config_dict).to(device)
# loss_function = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
nsp_loss_function = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
mlm_loss_function = torch.nn.CrossEntropyLoss(ignore_index=config_dict['pad_idx'], reduction='mean').to(device) #v2
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9,0.999), weight_decay=0.01, eps=1e-6)
print("Model READY !!!")

writer = SummaryWriter(log_dir=f"./runs/{current_name}")
st_time = time.time()
epoch = 0
step = 0
iteration = 0
flag_batch_num = 0
train_total_loss = 0
train_nsp_loss = 0
train_mlm_loss = 0
train_stop_flag = False

while True:
    pretrain_data_number = (epoch % 6)
    print(f"Preparing {pretrain_data_number} data...")
    pretraining_dataset_file_path = f"pretraining_{pretrain_data_number}.json"
    pretraining_dataset = PretrainingCustomDataset(pretraining_dataset_file_path, 1, 0, pad_idx)
    train_dataloader = DataLoader(pretraining_dataset, batch_size)
    epoch += 1
    for tokens, segment_ids, is_next, masked_lm_positions, masked_lm_labels in train_dataloader:
        flag_batch_num += batch_size
        
        tokens = tokens.to(device)
        segment_ids = segment_ids.to(device)
        # position_ids = position_ids.to(device)
        is_next = is_next.to(device)
        masked_lm_positions = masked_lm_positions.to(device)
        masked_lm_labels = masked_lm_labels.to(device)
        
        model.train()
        nsp_predict, mlm_predict = model.forward(tokens, segment_ids, masked_lm_positions)
        nsp_loss = nsp_loss_function(nsp_predict, is_next)
        mlm_loss = mlm_loss_function(mlm_predict, masked_lm_labels.reshape(-1))
        total_loss = nsp_loss + mlm_loss
        total_loss.backward()
        iteration += 1
        train_nsp_loss += nsp_loss.detach().cpu().item()
        train_mlm_loss += mlm_loss.detach().cpu().item()
        # train_total_loss += total_loss.detach().cpu().item()
        
        if step_batch_size <= flag_batch_num:
            flag_batch_num = 0
            
            if step <= warmup_steps:
                optimizer.param_groups[0]['lr'] = (learning_rate)*(step/warmup_steps)
            else:
                optimizer.param_groups[0]['lr'] = (learning_rate)*(1-((step-warmup_steps)/(num_total_steps-warmup_steps)))
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # lr_update.step()
            optimizer.zero_grad()
            step += 1
            
            if step % 1000 == 0:
                train_nsp_loss /= iteration
                train_mlm_loss /= iteration
                train_total_loss = train_nsp_loss + train_mlm_loss
                iteration = 0
                print(f"Step: {epoch}/{step:<8} lr: {optimizer.param_groups[0]['lr']:<9.1e} Train NSP Loss: {train_nsp_loss:<8.4f} Train MLM Loss: {train_mlm_loss:<8.4f} Train Loss: {train_total_loss:<8.4f} Time:{(time.time()-st_time)/3600:>6.4f} Hour")
                writer.add_scalars("nsp", {"nsp_loss":train_nsp_loss}, step)
                writer.add_scalars("mlm", {"mlm_loss":train_mlm_loss}, step)
                writer.add_scalars("total", {"total_loss":train_total_loss}, step)
                writer.flush()
                train_nsp_loss = 0
                train_mlm_loss = 0
                train_total_loss = 0
                
            if step == num_total_steps:
                train_stop_flag = True
                break
            
        if step % 10000 == 0:
            torch.save({'epoch': epoch,
                        'step': step,
                        'model': model,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, f"./save_model/{current_name}_CheckPoint.pth")
    if train_stop_flag:
        torch.save({'epoch': epoch,
                    'step': step,
                    'model': model,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, f"./save_model/{current_name}_CheckPoint.pth")
        break
    
print("="*50, sep="")
print("TRAINING FINISH", sep="")
print("="*50, sep="")