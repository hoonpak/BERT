import os
import json
import time
import torch
from torch.utils.data import IterableDataset, Dataset

class PretrainingCustomDataset(IterableDataset):
    def __init__(self, file_path, world_size, rank, pad_idx):
        self.file_path = file_path
        self.world_size = world_size
        self.rank = rank
        self.pad_idx = pad_idx
        st_time = time.time()
        self.total_lines = self._get_total_lines()
        print(f"Calculate length of line... : {self.total_lines} ", time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-st_time)))
        
    def _get_total_lines(self):
        with open(self.file_path, 'r') as f:
            return sum(1 for _ in f)

    def _reset(self):
        self.file = open(self.file_path, 'r')
    
    def _set_line_offset(self, start_line):
        self._reset()
        for _ in range(start_line):
            self.file.readline() 
    
    def __iter__(self):
        lines_per_rank = self.total_lines // self.world_size
        start_line = self.rank * lines_per_rank
        end_line = None if self.rank == self.world_size - 1 else (self.rank + 1) * lines_per_rank

        self._set_line_offset(start_line)
        self.end_line = end_line
        self.current_line = start_line

        return self

    def __next__(self):
        if self.end_line is not None and self.current_line >= self.end_line:
            self.file.close()
            raise StopIteration

        line = self.file.readline()
        if not line:
            self.file.close()
            raise StopIteration

        self.current_line += 1

        # JSON 형식으로 파싱
        data = json.loads(line.strip())

        tokens = data['tokens']
        segment_ids = data['segment_ids']
        is_random_next = data['is_random_next']
        masked_lm_positions = data['masked_lm_positions']
        masked_lm_labels = data['masked_lm_labels']
        
        real_sen_token_num = len(tokens)
        pad_mul_num = 128 - real_sen_token_num
        
        # position_ids = torch.arange(real_sen_token_num)
        # pad_dim = (0, pad_mul_num)
        # position_ids = torch.nn.functional.pad(position_ids, pad_dim, "constant", 128)
        
        tokens = tokens + [self.pad_idx]*pad_mul_num
        segment_ids = segment_ids + [2]*pad_mul_num
        
        # masked_pad_list = [self.pad_idx]*(20 - len(masked_lm_positions))
        masked_lm_positions = masked_lm_positions + [-1]*(20 - len(masked_lm_positions))
        masked_lm_labels = masked_lm_labels + [self.pad_idx]*(20 - len(masked_lm_labels))

        is_next = 1
        if is_random_next:
            is_next = 0
        
        return [torch.LongTensor(tokens), torch.LongTensor(segment_ids), is_next,
                torch.LongTensor(masked_lm_positions), torch.LongTensor(masked_lm_labels)]
        
class FinetuningCustomDataset(Dataset):
    def __init__(self, dataname, extracted_data, label_encoder, cls_token_id, sep_token_id, pad_token_id):
        self.dataname = dataname
        self.train_data = extracted_data
        self.label_encoder = label_encoder
        self.cls_token_id = [cls_token_id]
        self.sep_token_id = [sep_token_id]
        self.pad_token_id = [pad_token_id]
        
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, index):
        data = self.train_data[index]
        if len(data) == 2:
            sentence, label = data
            train_data = self.cls_token_id + sentence + self.sep_token_id
            segment_id = [0]*len(train_data)
        elif len(data) == 3:
            sentence_a, sentence_b, label = data
            train_data_a = self.cls_token_id + sentence_a + self.sep_token_id
            train_data_b = sentence_b + self.sep_token_id
            segment_id = [0]*len(train_data_a) + [1]*len(train_data_b)
            train_data = train_data_a + train_data_b
        else:
            raise ValueError("There is something wrong with data shape or size. That should be have 2 or 3 argument.")

        pad_length = 128 - len(train_data)
        train_data = train_data + pad_length*self.pad_token_id
        segment_id = segment_id + [2]*pad_length
        
        if self.label_encoder:
            label = self.label_encoder[label]
        
        if self.dataname == "STS-B":
            label = torch.FloatTensor([label])
        else:
            label = int(label)
        
        return [torch.LongTensor(train_data), torch.LongTensor(segment_id), label]

class GetDataFromFile:
    def __init__(self, data_name, file_path, tokenizer):
        self.tokenizer = tokenizer
        self.label_encoder = None
        
        if data_name == 'mnli':
            mnli_path_list = os.listdir(file_path)
            train_data_path, = [os.path.join(file_path, name) for name in mnli_path_list if "train" in name]
            dev_mat_data_path, = [os.path.join(file_path, name) for name in mnli_path_list if "dev_mat" in name]
            dev_mis_data_path, = [os.path.join(file_path, name) for name in mnli_path_list if "dev_mis" in name]
            
            self.train_extracted_data, self.label_encoder = self.get_MNLI(train_data_path)
            self.dev_mat_extracted_data, dev_mat_label_encoder = self.get_MNLI(dev_mat_data_path)
            self.dev_mis_extracted_data, dev_mis_label_encoder = self.get_MNLI(dev_mis_data_path)
                
        elif data_name == 'wnli':
            path_list = os.listdir(file_path)
            train_data_path, = [os.path.join(file_path, name) for name in path_list if "train" in name]
            dev_data_path, = [os.path.join(file_path, name) for name in path_list if "dev" in name]
            self.train_extracted_data = self.get_WNLI(train_data_path)
            self.dev_extracted_data = self.get_WNLI(dev_data_path)
            
        elif data_name == 'cola':
            path_list = os.listdir(file_path)
            train_data_path, = [os.path.join(file_path, name) for name in path_list if "train" in name]
            dev_data_path, = [os.path.join(file_path, name) for name in path_list if "dev" in name]
            self.train_extracted_data = self.get_CoLA(train_data_path)
            self.dev_extracted_data = self.get_CoLA(dev_data_path)
            
        elif data_name == 'mrpc':
            path_list = os.listdir(file_path)
            train_data_path, = [os.path.join(file_path, name) for name in path_list if "train" in name]
            dev_data_path, = [os.path.join(file_path, name) for name in path_list if "_test" in name]
            self.train_extracted_data = self.get_MRPC(train_data_path)
            self.dev_extracted_data = self.get_MRPC(dev_data_path)
        
        elif data_name == 'rte':
            path_list = os.listdir(file_path)
            train_data_path, = [os.path.join(file_path, name) for name in path_list if "train" in name]
            dev_data_path, = [os.path.join(file_path, name) for name in path_list if "dev" in name]
            
            self.train_extracted_data, self.label_encoder = self.get_RTE(train_data_path)
            self.dev_extracted_data, dev_label_encoder = self.get_RTE(dev_data_path)
        
        elif data_name == 'qqp':
            path_list = os.listdir(file_path)
            train_data_path, = [os.path.join(file_path, name) for name in path_list if "train" in name]
            dev_data_path, = [os.path.join(file_path, name) for name in path_list if "dev" in name]
            self.train_extracted_data = self.get_QQP(train_data_path)
            self.dev_extracted_data = self.get_QQP(dev_data_path)

        elif data_name == 'sst-2':
            path_list = os.listdir(file_path)
            train_data_path, = [os.path.join(file_path, name) for name in path_list if "train" in name]
            dev_data_path, = [os.path.join(file_path, name) for name in path_list if "dev" in name]
            self.train_extracted_data = self.get_SST2(train_data_path)
            self.dev_extracted_data = self.get_SST2(dev_data_path)

        elif data_name == 'qnli':
            path_list = os.listdir(file_path)
            train_data_path, = [os.path.join(file_path, name) for name in path_list if "train" in name]
            dev_data_path, = [os.path.join(file_path, name) for name in path_list if "dev" in name]
            
            self.train_extracted_data, self.label_encoder = self.get_QNLI(train_data_path)
            self.dev_extracted_data, dev_label_encoder = self.get_QNLI(dev_data_path)

        elif data_name == 'sts-b':
            path_list = os.listdir(file_path)
            train_data_path, = [os.path.join(file_path, name) for name in path_list if "train" in name]
            dev_data_path, = [os.path.join(file_path, name) for name in path_list if "dev" in name]
            self.train_extracted_data = self.get_STSB(train_data_path)
            self.dev_extracted_data = self.get_STSB(dev_data_path)
        
                                
    def get_MNLI(self, file_path):
        with open(file_path, "r") as file:
            data_lines = file.readlines()
        extracted_data = []
        label_encoder = dict()
        idx = 0
        for data_line in data_lines[1:]:
            data_line_list = data_line.strip().split("\t")
            sentence_a = self.tokenizer.encode(data_line_list[8]).ids
            if len(sentence_a) > 125:
                continue
            sentence_b = self.tokenizer.encode(data_line_list[9]).ids
            if len(sentence_a) + len(sentence_b) > 125:
                continue
            class_label = data_line_list[-1]
            if class_label not in label_encoder.keys():
                label_encoder[class_label] = idx
                idx += 1
            extracted_data.append((sentence_a, sentence_b, class_label))
        return extracted_data, label_encoder

    def get_WNLI(self, file_path):
        with open(file_path, "r") as file:
            data_lines = file.readlines()
        extracted_data = []
        for data_line in data_lines[1:]:
            data_line_list = data_line.strip().split("\t")
            sentence_a = self.tokenizer.encode(data_line_list[1]).ids
            if len(sentence_a) > 125:
                continue
            sentence_b = self.tokenizer.encode(data_line_list[2]).ids
            if len(sentence_a) + len(sentence_b) > 125:
                continue
            class_label = data_line_list[-1]
            extracted_data.append((sentence_a, sentence_b, class_label))
        return extracted_data
    
    def get_CoLA(self, file_path):
        with open(file_path, "r") as file:
            data_lines = file.readlines()
        extracted_data = []
        for data_line in data_lines:
            data_line_list = data_line.strip().split("\t")
            sentence = self.tokenizer.encode(data_line_list[-1]).ids
            if len(sentence) > 125:
                continue
            class_label = data_line_list[1]
            extracted_data.append((sentence, class_label))
        return extracted_data

    def get_MRPC(self, file_path):
        with open(file_path, "r") as file:
            data_lines = file.readlines()
        extracted_data = []
        for data_line in data_lines[1:]:
            data_line_list = data_line.strip().split("\t")
            sentence_a = self.tokenizer.encode(data_line_list[3]).ids
            if len(sentence_a) > 125:
                continue
            sentence_b = self.tokenizer.encode(data_line_list[4]).ids
            if len(sentence_a) + len(sentence_b) > 125:
                continue
            class_label = data_line_list[0]
            extracted_data.append((sentence_a, sentence_b, class_label))
        return extracted_data

    def get_RTE(self, file_path):
        with open(file_path, "r") as file:
            data_lines = file.readlines()
        extracted_data = []
        label_encoder = dict()
        idx = 0
        for data_line in data_lines[1:]:
            data_line_list = data_line.strip().split("\t")
            sentence_a = self.tokenizer.encode(data_line_list[1]).ids
            if len(sentence_a) > 125:
                continue
            sentence_b = self.tokenizer.encode(data_line_list[2]).ids
            if len(sentence_a) + len(sentence_b) > 125:
                continue
            class_label = data_line_list[-1]
            if class_label not in label_encoder.keys():
                label_encoder[class_label] = idx
                idx += 1
            extracted_data.append((sentence_a, sentence_b, class_label))
        return extracted_data, label_encoder

    def get_QQP(self, file_path):
        with open(file_path, "r") as file:
            data_lines = file.readlines()
        extracted_data = []
        for data_line in data_lines[1:]:
            data_line_list = data_line.strip().split("\t")
            sentence_a = self.tokenizer.encode(data_line_list[3]).ids
            if len(sentence_a) > 125:
                continue
            sentence_b = self.tokenizer.encode(data_line_list[4]).ids
            if len(sentence_a) + len(sentence_b) > 125:
                continue
            class_label = data_line_list[-1]
            extracted_data.append((sentence_a, sentence_b, class_label))
        return extracted_data

    def get_SST2(self, file_path):
        with open(file_path, "r") as file:
            data_lines = file.readlines()
        extracted_data = []
        for data_line in data_lines[1:]:
            data_line_list = data_line.strip().split("\t")
            sentence = self.tokenizer.encode(data_line_list[0]).ids
            if len(sentence) > 125:
                continue
            class_label = data_line_list[-1]
            extracted_data.append((sentence, class_label))
        return extracted_data

    def get_QNLI(self, file_path):
        with open(file_path, "r") as file:
            data_lines = file.readlines()
        extracted_data = []
        label_encoder = dict()
        idx = 0
        for data_line in data_lines[1:]:
            data_line_list = data_line.strip().split("\t")
            sentence_a = self.tokenizer.encode(data_line_list[1]).ids
            if len(sentence_a) > 125:
                continue
            sentence_b = self.tokenizer.encode(data_line_list[2]).ids
            if len(sentence_a) + len(sentence_b) > 125:
                continue
            class_label = data_line_list[-1]
            if class_label not in label_encoder.keys():
                label_encoder[class_label] = idx
                idx += 1
            extracted_data.append((sentence_a, sentence_b, class_label))
        return extracted_data, label_encoder
    
    def get_STSB(self, file_path):
        with open(file_path, "r") as file:
            data_lines = file.readlines()
        extracted_data = []
        for data_line in data_lines[1:]:
            data_line_list = data_line.strip().split("\t")
            sentence_a = self.tokenizer.encode(data_line_list[-3]).ids
            if len(sentence_a) > 125:
                continue
            sentence_b = self.tokenizer.encode(data_line_list[-2]).ids
            if len(sentence_a) + len(sentence_b) > 125:
                continue
            class_label = float(data_line_list[-1])
            extracted_data.append((sentence_a, sentence_b, class_label))
        return extracted_data


