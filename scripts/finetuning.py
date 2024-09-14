import sys, os
sys.path.append("../src/")
from copy import deepcopy
import argparse
import time
import pickle

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from sklearn.metrics import f1_score, accuracy_score

from data import FinetuningCustomDataset, GetDataFromFile
from config import *
from model import ClassifierBERT

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="tiny", choices=["tiny", "small"])
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--version")
option = parser.parse_args()

device = option.device

config_dict = tiny_config
if option.model == "small":
    config_dict = small_config

current_name = option.model + "_" + option.version

data_path_names = ['SST-2', 'MRPC', "QQP", "QNLI", "RTE", "WNLI"]

dataset_path = "/root/BERT/dataset/glue_data"

tokenizer_file_path = "../dataset/BERT_Tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_file_path)

model_info = torch.load(f"../scripts/save_model/{current_name}_CheckPoint.pth", map_location="cpu")
model = model_info['model']
model.load_state_dict(model_info['model_state_dict'])

max_epochs = 4
batch_sizes = [8, 16, 32, 64, 128]
learning_rates = [3e-4, 1e-4, 5e-5, 3e-5]
cases = []
for b in batch_sizes:
    for lr in learning_rates:
        cases.append((b, lr))
        
st_time = time.time()
results = dict()
for dataname in data_path_names:
    print("#"*100,f"{dataname} TRAINING START!!!!!","#"*100)
    sub_results = dict()
    data_path = os.path.join(dataset_path, dataname)
    data_instance = GetDataFromFile(dataname.lower(), data_path, tokenizer)
    training_dataset = FinetuningCustomDataset(data_instance.train_extracted_data, data_instance.label_encoder,
                                              tokenizer.token_to_id("[CLS]"), tokenizer.token_to_id("[SEP]"), tokenizer.token_to_id("[PAD]"))
    test_dataset = FinetuningCustomDataset(data_instance.dev_extracted_data, data_instance.label_encoder,
                                              tokenizer.token_to_id("[CLS]"), tokenizer.token_to_id("[SEP]"), tokenizer.token_to_id("[PAD]"))
    _,cls_list = zip(*data_instance.train_extracted_data)
    cls_num = len(set(cls_list))
    
    for batch_size, learning_rate in cases:
        sub_key = str(batch_size) + "_" + str(learning_rate)
        print("$"*105,f"{sub_key} case...","$"*105)
        training_dataloader = DataLoader(training_dataset, batch_size, True)
        test_dataloader = DataLoader(test_dataset, batch_size, False)

        bert = deepcopy(model.bert_layer)
        clsmodel = ClassifierBERT(config_dict, bert, cls_num).to(device)
        loss_function = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(clsmodel.parameters(), lr=learning_rate, betas=(0.9,0.999), weight_decay=0.01)

        train_loss = 0
        train_acc = 0
        
        for epoch in range(max_epochs):
            clsmodel.train()
            for input, segment, label  in training_dataloader:
                input = input.to(device)
                segment = segment.to(device)
                label = label.to(device)
                
                predict = clsmodel(input, segment)
                loss = loss_function(predict, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                train_loss += loss.detach().cpu().item()                
                correct = sum(predict.max(dim=-1)[1] == label).item()
                train_acc += correct
                
            train_acc*=(100/len(training_dataset))
            train_loss*=(batch_size/len(training_dataset))
            cur_time = time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-st_time))
            print(f"Data: {dataname:<6} Epoch: {epoch:<2} batch size: {batch_size:<3} lr: {optimizer.param_groups[0]['lr']:<5.1e} Train Loss: {train_loss:<5.2f} Train acc: {train_acc:<6.2f} Time:{cur_time}")
            train_acc = 0
            train_loss = 0
        
        clsmodel.eval()
        with torch.no_grad():
            preds = []
            labels = []
            for input, segment, label in test_dataloader:
                input = input.to(device)
                segment = segment.to(device)
                label = label.to(device)
                
                predict = clsmodel(input, segment)
                _, pred = predict.max(dim=-1)
                
                preds.extend(pred.cpu().numpy())
                labels.extend(label.cpu().numpy())
            test_f1 = f1_score(labels, preds, average='macro')
            test_f1 *= 100
            test_acc = accuracy_score(labels, preds)
            test_acc *= 100
            print("="*50, end="")
            cur_time = time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-st_time))
            print(f"Data: {dataname:<6} batch size: {batch_size:<3} lr: {optimizer.param_groups[0]['lr']:<5.1e} Test acc: {test_acc:<6.2f} Test f1: {test_f1:<6.2f} Time:{cur_time}", end="")
            print("="*50)
        
        sub_results[sub_key] = (test_acc, test_f1)
    results[dataname] = sub_results

print("="*210)
print("="*200, end="")
print("FINISH", end="")
print("="*200)
print("="*210)

with open(f'{current_name}_finetuned_results.pkl', 'wb') as f:
    pickle.dump(results, f)