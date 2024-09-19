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
    print("="*50, end="")
    cur_time = time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-st_time))
    print(f"Data: {dataname:<6} batch size: {batch_size:<3} lr: {optimizer.param_groups[0]['lr']:<5.1e} Test acc: {test_acc:<6.2f} Test f1: {test_f1:<6.2f} Time:{cur_time}", end="")
    print("="*50)
    return test_acc, test_f1


if __name__ == "__main__":

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

    # data_path_names = ['SST-2', 'MRPC', "QQP", "QNLI", "RTE", "WNLI"]
    # data_path_names = ['SST-2', 'MRPC', "QQP", "QNLI", 'STS-B']
    data_path_names = ["RTE", "WNLI",'MNLI']
    # data_path_names = ["QQP", "QNLI", "RTE", "WNLI"]
    # data_path_names = ['MNLI']
    # data_path_names = ['STS-B']

    dataset_path = "../dataset/glue_data"

    tokenizer_file_path = "../dataset/BERT_Tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_file_path)

    model_info = torch.load(f"../scripts/save_model/{current_name}_CheckPoint.pth", map_location="cpu")
    model = model_info['model']
    model.load_state_dict(model_info['model_state_dict'])

    max_epochs = 4
    batch_sizes = [8, 16, 32, 64, 128]
    learning_rates = [3e-4, 1e-4, 5e-5, 3e-5]
    # learning_rates = [3e-4, 1e-4, 5e-5, 3e-5, 2e-5]
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
        
        if dataname == "MNLI":
            training_dataset = FinetuningCustomDataset(dataname, data_instance.train_extracted_data, data_instance.label_encoder,
                                                    tokenizer.token_to_id("[CLS]"), tokenizer.token_to_id("[SEP]"), tokenizer.token_to_id("[PAD]"))
            m_test_dataset = FinetuningCustomDataset(dataname, data_instance.dev_mat_extracted_data, data_instance.label_encoder,
                                                    tokenizer.token_to_id("[CLS]"), tokenizer.token_to_id("[SEP]"), tokenizer.token_to_id("[PAD]"))
            mm_test_dataset = FinetuningCustomDataset(dataname, data_instance.dev_mis_extracted_data, data_instance.label_encoder,
                                                    tokenizer.token_to_id("[CLS]"), tokenizer.token_to_id("[SEP]"), tokenizer.token_to_id("[PAD]"))
        else:
            training_dataset = FinetuningCustomDataset(dataname, data_instance.train_extracted_data, data_instance.label_encoder,
                                                    tokenizer.token_to_id("[CLS]"), tokenizer.token_to_id("[SEP]"), tokenizer.token_to_id("[PAD]"))
            test_dataset = FinetuningCustomDataset(dataname, data_instance.dev_extracted_data, data_instance.label_encoder,
                                                tokenizer.token_to_id("[CLS]"), tokenizer.token_to_id("[SEP]"), tokenizer.token_to_id("[PAD]"))
        
        if len(data_instance.train_extracted_data[0]) == 2:
            _,cls_list = zip(*data_instance.train_extracted_data)
        elif len(data_instance.train_extracted_data[0]) == 3:
            _,_,cls_list = zip(*data_instance.train_extracted_data)
        else:
            raise ValueError("check!")
        cls_num = len(set(cls_list))
        if dataname == "STS-B":
            cls_num = 1
            
        for batch_size, learning_rate in cases:
            sub_key = str(batch_size) + "_" + str(learning_rate)
            print("$"*105,f"{sub_key} case...","$"*105)
            training_dataloader = DataLoader(training_dataset, batch_size, True)
            if dataname == "MNLI":
                m_test_dataloader = DataLoader(m_test_dataset, batch_size, False)        
                mm_test_dataloader = DataLoader(mm_test_dataset, batch_size, False)        
            else:
                test_dataloader = DataLoader(test_dataset, batch_size, False)

            bert = deepcopy(model.bert_layer)
            clsmodel = ClassifierBERT(config_dict, bert, cls_num).to(device)
            loss_function = torch.nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.Adam(clsmodel.parameters(), lr=learning_rate)

            train_loss = 0
            train_acc = 0
            
            if dataname == "STS-B":
                loss_function = torch.nn.MSELoss().to(device)
                for epoch in range(max_epochs):
                    clsmodel.train()
                    for input, segment, label  in training_dataloader:
                        input = input.to(device)
                        segment = segment.to(device)
                        label = label.to(device)
                        
                        predict = clsmodel(input, segment)
                        # predict = 5*torch.sigmoid(predict)
                        loss = loss_function(predict, label)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        train_loss += loss.detach().cpu().item()                
                        
                    train_loss*=(batch_size/len(training_dataset))
                    cur_time = time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-st_time))
                    print(f"Data: {dataname:<6} Epoch: {epoch:<2} batch size: {batch_size:<3} lr: {optimizer.param_groups[0]['lr']:<5.1e} Train Loss: {train_loss:<5.2f} Time:{cur_time}")
                    train_loss = 0
                    
            else:
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
                    # clsmodel.eval()
                    # with torch.no_grad():
                    #     test_acc, test_f1 = test_acc_f1(test_dataloader, dataname, batch_size, optimizer, st_time)

            
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

                        preds.extend(pred.detach().cpu().numpy())
                        labels.extend(label.cpu().numpy())
                    preds = [x for xx in preds for x in xx]
                    labels = [x for xx in labels for x in xx]
                    pearson,_ = pearsonr(labels, preds)
                    pearson *= 100
                    spearman,_ = spearmanr(labels, preds)
                    spearman *= 100
                    print("="*50, end="")
                    cur_time = time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-st_time))
                    print(f"Data: {dataname:<6} batch size: {batch_size:<3} lr: {optimizer.param_groups[0]['lr']:<5.1e} Test pearson: {pearson:<6.2f} Test spearman: {spearman:<6.2f} Time:{cur_time}", end="")
                    print("="*50)
                    test_acc = pearson
                    test_f1 = spearman
                else:
                    test_acc, test_f1 = test_acc_f1(test_dataloader, dataname, batch_size, optimizer, st_time)
            
            sub_results[sub_key] = (test_acc, test_f1)
        results[dataname] = sub_results

    print("#"*210)
    print("#"*100, end="")
    print("FINISH", end="")
    print("#"*100)
    print("#"*210)

    with open(f'1{current_name}_finetuned_results.pkl', 'wb') as f:
        pickle.dump(results, f)