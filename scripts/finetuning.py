import sys, os
sys.path.append("../src/")
from copy import deepcopy
import argparse
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
# from tokenizers import Tokenizer
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from data import FinetuningCustomDataset, GetDataFromFile
from config import *
from model import ClassifierBERT

def test_acc_f1(test_dataloader, dataname, batch_size, optimizer, st_time):
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

    cur_time = time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-st_time))
    print(f"Data: {dataname:<6} batch size: {batch_size:<3} lr: {optimizer.param_groups[0]['lr']:<7.1e} Test acc: {test_acc:<6.2f} Test f1: {test_f1:<6.2f} Time:{cur_time}")

    return test_acc, test_f1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny", choices=["tiny", "small"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--version")
    parser.add_argument("--name")
    option = parser.parse_args()

    device = option.device

    config_dict = tiny_config
    if option.model == "small":
        config_dict = small_config

    current_name = option.model + "_" + option.version

    # data_path_names = ["SST-2", "MRPC", "STS-B", "QQP", "MNLI", "QNLI", "RTE", "WNLI", "CoLA"]
    if option.name == "0":
        data_path_names = ["CoLA", "SST-2", "MRPC", "STS-B"]
    elif option.name == "1":
        data_path_names = ["QQP"]
    else:
        data_path_names = ["MNLI", "QNLI", "RTE", "WNLI"]
        # data_path_names = ["CoLA"]

    dataset_path = "../dataset/glue_data"

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', use_fast=True)
    pad_idx = tokenizer.convert_tokens_to_ids("[PAD]")
    
    model_info = torch.load(f"../scripts/save_model/{current_name}_CheckPoint.pth", map_location="cpu")
    model = model_info['model']
    model.load_state_dict(model_info['model_state_dict'])
    
    # for param in model.bert_layer.parameters():
    #     param.requires_grad = False
    # for param in model.bert_layer.pooling_layer.parameters():
    #     param.requires_grad = True

    max_epochs = 4
    batch_sizes = [8, 16, 32, 64, 128]
    # batch_sizes = [8, 16, 32]
    # batch_sizes = [64, 128]
    learning_rates = [3e-4, 1e-4, 5e-5, 3e-5]
    # max_epochs = 100
    # batch_sizes = [128]
    # learning_rates = [3e-4]

    cases = []
    for b in batch_sizes:
        for lr in learning_rates:
            cases.append((b, lr))
            
    st_time = time.time()
    results = dict()
    for dataname in data_path_names:
        print("#"*50,f"{dataname} TRAINING START!!!!!","#"*50)
        sub_results = dict()
        data_path = os.path.join(dataset_path, dataname)
        data_instance = GetDataFromFile(dataname.lower(), data_path, tokenizer)
        
        if dataname == "MNLI":
            training_dataset = FinetuningCustomDataset(dataname, data_instance.train_extracted_data, data_instance.label_encoder, 
                                                       tokenizer.convert_tokens_to_ids("[CLS]"), tokenizer.convert_tokens_to_ids("[SEP]"), pad_idx) 
            m_test_dataset = FinetuningCustomDataset(dataname, data_instance.dev_mat_extracted_data, data_instance.label_encoder, 
                                                     tokenizer.convert_tokens_to_ids("[CLS]"), tokenizer.convert_tokens_to_ids("[SEP]"), pad_idx)
            mm_test_dataset = FinetuningCustomDataset(dataname, data_instance.dev_mis_extracted_data, data_instance.label_encoder, 
                                                      tokenizer.convert_tokens_to_ids("[CLS]"), tokenizer.convert_tokens_to_ids("[SEP]"), pad_idx)
        else:
            training_dataset = FinetuningCustomDataset(dataname, data_instance.train_extracted_data, data_instance.label_encoder, 
                                                       tokenizer.convert_tokens_to_ids("[CLS]"), tokenizer.convert_tokens_to_ids("[SEP]"), pad_idx)
            test_dataset = FinetuningCustomDataset(dataname, data_instance.dev_extracted_data, data_instance.label_encoder, 
                                                   tokenizer.convert_tokens_to_ids("[CLS]"), tokenizer.convert_tokens_to_ids("[SEP]"), pad_idx)
        
        if len(data_instance.train_extracted_data[0]) == 2:
            _,cls_list = zip(*data_instance.train_extracted_data)
        elif len(data_instance.train_extracted_data[0]) == 3:
            _,_,cls_list = zip(*data_instance.train_extracted_data)
        else:
            raise ValueError("check!")
        
        cls_num = len(set(cls_list))
        if dataname == "STS-B":
            cls_num=1
            
        for batch_size, learning_rate in cases:
            sub_key = str(batch_size) + "_" + str(learning_rate)
            print("$"*43,f"{sub_key} case...","$"*43)
            training_dataloader = DataLoader(training_dataset, batch_size, True)
            if dataname == "MNLI":
                m_test_dataloader = DataLoader(m_test_dataset, batch_size, False)        
                mm_test_dataloader = DataLoader(mm_test_dataset, batch_size, False)        
            else:
                test_dataloader = DataLoader(test_dataset, batch_size, False)

            bert = deepcopy(model.bert_layer)
            clsmodel = ClassifierBERT(config_dict, bert, cls_num).to(device)
            optimizer = torch.optim.AdamW(clsmodel.parameters(), lr=learning_rate, eps=1e-6, weight_decay=0.01)
            num_total_steps = training_dataset.__len__()*max_epochs
            warmup_steps = round(num_total_steps*(0.1))
            
            if dataname == "STS-B":
                loss_function = torch.nn.L1Loss().to(device)
                train_loss = 0
                step = 0
                for epoch in range(max_epochs):
                    clsmodel.train()
                    for input, segment, label  in training_dataloader:
                        step += 1
                        input = input.to(device)
                        segment = segment.to(device)
                        label = label.to(device)
                        
                        predict = clsmodel(input, segment)
                        loss = loss_function(predict, label)
                        loss.backward()
                        if step <= warmup_steps:
                            optimizer.param_groups[0]['lr'] = (learning_rate)*(step/warmup_steps)
                        else:
                            proportion = 1-((step-warmup_steps)/(num_total_steps-warmup_steps))
                            if proportion <= 0:
                                optimizer.param_groups[0]['lr'] = 1e-7
                            else:
                                optimizer.param_groups[0]['lr'] = (learning_rate)*proportion
                        
                        torch.nn.utils.clip_grad_norm_(clsmodel.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        train_loss += loss.detach().cpu().item()                
                    
                    train_loss*=(batch_size/len(training_dataset))
                    cur_time = time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-st_time))
                    print(f"Data: {dataname:<6} Epoch: {epoch:<2} batch size: {batch_size:<3} lr: {optimizer.param_groups[0]['lr']:<7.1e} Train Loss: {train_loss:<5.2f} Time:{cur_time}")
                    train_loss = 0

            else:
                loss_function = torch.nn.CrossEntropyLoss().to(device)

                train_loss = 0
                train_acc = 0
                step = 0
                
                for epoch in range(max_epochs):
                    clsmodel.train()
                    for input, segment, label  in training_dataloader:
                        step += 1
                        input = input.to(device)
                        segment = segment.to(device)
                        label = label.to(device)
                        
                        predict = clsmodel(input, segment)
                        loss = loss_function(predict, label)
                        loss.backward()
                        if step <= warmup_steps:
                            optimizer.param_groups[0]['lr'] = (learning_rate)*(step/warmup_steps)
                        else:
                            proportion = 1-((step-warmup_steps)/(num_total_steps-warmup_steps))
                            if proportion <= 0:
                                optimizer.param_groups[0]['lr'] = 1e-7
                            else:
                                optimizer.param_groups[0]['lr'] = (learning_rate)*proportion
                        
                        torch.nn.utils.clip_grad_norm_(clsmodel.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        train_loss += loss.detach().cpu().item()                
                        correct = sum(predict.max(dim=-1)[1] == label).item()
                        train_acc += correct
                        
                    train_acc*=(100/len(training_dataset))
                    train_loss*=(batch_size/len(training_dataset))
                    cur_time = time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-st_time))
                    print(f"Data: {dataname:<6} Epoch: {epoch:<2} batch size: {batch_size:<3} lr: {optimizer.param_groups[0]['lr']:<7.1e} Train Loss: {train_loss:<5.2f} Train acc: {train_acc:<6.2f} Time:{cur_time}")
                    train_acc = 0
                    train_loss = 0

            clsmodel.eval()
            with torch.no_grad():
                if dataname == "MNLI":
                    m_test_acc, m_test_f1 = test_acc_f1(m_test_dataloader, dataname, batch_size, optimizer, st_time)
                    mm_test_acc, mm_test_f1 = test_acc_f1(mm_test_dataloader, dataname, batch_size, optimizer, st_time)
                    test_acc = (m_test_acc, mm_test_acc)
                    test_f1 = (m_test_f1, mm_test_f1)
                elif dataname == "STS-B":
                    preds = []
                    labels = []
                    for input, segment, label in test_dataloader:
                        input = input.to(device)
                        segment = segment.to(device)
                        label = label.to(device)
                        
                        pred = clsmodel(input, segment)
                        preds.extend(pred.cpu().numpy())
                        labels.extend(label.cpu().numpy())
                        
                    preds = [x for xx in preds for x in xx]
                    labels = [x for xx in labels for x in xx]
                    pearson,_ = pearsonr(labels, preds)
                    pearson = pearson*50+50
                    spearman,_ = spearmanr(labels, preds)
                    spearman = spearman*50+50
                    cur_time = time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-st_time))
                    print(f"Data: {dataname:<6} batch size: {batch_size:<3} lr: {optimizer.param_groups[0]['lr']:<7.1e} Test pearson: {pearson:<6.2f} Test spearman: {spearman:<6.2f} Time:{cur_time}")
                    test_acc = pearson
                    test_f1 = spearman
                elif dataname == "CoLA":
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
                        
                        predict = clsmodel(input, segment)
                        
                    matthews = matthews_corrcoef(labels, preds)
                    matthews = matthews*100
                    cur_time = time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-st_time))
                    print(f"Data: {dataname:<6} batch size: {batch_size:<3} lr: {optimizer.param_groups[0]['lr']:<7.1e} Test matthews: {matthews:<6.2f} Time:{cur_time}")
                    test_acc = matthews
                    test_f1 = matthews
                else:
                    test_acc, test_f1 = test_acc_f1(test_dataloader, dataname, batch_size, optimizer, st_time)
            
            sub_results[sub_key] = (test_acc, test_f1)
        results[dataname] = sub_results

    print("#"*106)
    print("#"*50, end="")
    print("FINISH", end="")
    print("#"*50)
    print("#"*106)

    with open(f'{option.name}{current_name}_finetuned_results.pkl', 'wb') as f:
        pickle.dump(results, f)