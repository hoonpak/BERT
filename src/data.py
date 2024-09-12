import json
import torch
from torch.utils.data import IterableDataset

class PretrainingCustomDataset(IterableDataset):
    def __init__(self, file_path, world_size, rank, pad_idx):
        self.file_path = file_path
        self.world_size = world_size
        self.rank = rank
        self.pad_idx = pad_idx
        self.total_lines = self._get_total_lines()

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
        
        masked_pad_list = [self.pad_idx]*(20 - len(masked_lm_positions))
        masked_lm_positions = masked_lm_positions + masked_pad_list
        masked_lm_labels = masked_lm_labels + masked_pad_list

        is_next = 1
        if is_random_next:
            is_next = 0
        
        return [torch.LongTensor(tokens), torch.LongTensor(segment_ids), is_next,
                torch.LongTensor(masked_lm_positions), torch.LongTensor(masked_lm_labels)]
        
#torch.LongTensor(position_ids),