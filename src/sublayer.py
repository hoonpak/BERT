import torch
from torch import nn

from util import truncated_normal_

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        assert config['dim_model'] % config['num_attention_heads'] == 0
        self.WQ = nn.Linear(config['dim_model'], config['dim_model'], bias=True)
        self.WK = nn.Linear(config['dim_model'], config['dim_model'], bias=True)
        self.WV = nn.Linear(config['dim_model'], config['dim_model'], bias=True)
        self.WO = nn.Linear(config['dim_model'], config['dim_model'], bias=True)
        self.dropout = nn.Dropout(p=config['hidden_dropout_prob'])
        
        truncated_normal_(self.WQ.weight)
        self.WQ.bias.data.zero_()
        truncated_normal_(self.WK.weight)
        self.WK.bias.data.zero_()
        truncated_normal_(self.WV.weight)
        self.WV.bias.data.zero_()
        truncated_normal_(self.WO.weight)
        self.WO.bias.data.zero_()
        
        self.head = config['num_attention_heads']
        self.dim_model = config['dim_model']
        self.d_k = config['dim_model']//config['num_attention_heads']
        self.d_v = config['dim_model']//config['num_attention_heads']
        self.scaler = torch.sqrt(torch.tensor(self.d_k))
        
    def forward(self, Query, Key, Value, masked_info):
        """
        **INPUT SHAPE**
        masked_info - N, QL, L -> padding = True, else = False
        """
        QN, QL, QD = Query.shape
        KN, KL, KD = Key.shape
        VN, VL, VD = Value.shape
        
        query = self.WQ(Query).reshape(QN,QL,self.head,self.d_k).transpose(1,2) #N, L, D => N, L, H, D/H => N, H, L, D/H
        key = self.WK(Key).reshape(KN,KL,self.head,self.d_k).transpose(1,2) #N, L, D => N, L, H, D/H => N, H, L, D/H
        value = self.WV(Value).reshape(VN,VL,self.head,self.d_v).transpose(1,2) #N, L, D => N, L, H, D/H => N, H, L, D/H
        
        scaled_output = torch.matmul(query, key.transpose(-1,-2))/self.scaler #N, H, QL, D/H * N, H, D/H, L => N, H, QL, L
        
        # masked_info = masked_info.unsqueeze(1).repeat(1,self.head,1,1) #N, QL, L => N, H, QL, L
        attn_score = scaled_output.masked_fill(masked_info, float("-inf")).softmax(-1).nan_to_num(0) #N, H, QL, L
        attn_score = self.dropout(attn_score)
        multi_outputs = torch.matmul(attn_score, value).transpose(1,2).reshape(VN, QL, self.dim_model)  #N, H, QL, L * N, H, L, D/H => N, H, QL, D/H => N, QL, H, D/H => N, QL, D
        output = self.WO(multi_outputs) #N, QL, d_m
        return output