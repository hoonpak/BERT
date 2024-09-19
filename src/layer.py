import torch
from torch import nn

from util import truncated_normal_

class EmbeddingLayer(nn.Module):
    def __init__(self, config, sharing_embedding):
        super(EmbeddingLayer, self).__init__()
        self.tokens_embedding_layer = sharing_embedding
        self.segments_embedding_layer = nn.Embedding(num_embeddings=config['types'], embedding_dim=config['dim_model'], padding_idx=2)
        # self.positions_embedding_layer = nn.Embedding(num_embeddings=config['max_position_embeddings']+1, embedding_dim=config['dim_model'], padding_idx=128)
        self.positions_embedding = nn.Parameter(torch.rand(config['max_position_embeddings'], config['dim_model']))
        
        self.layer_norm = nn.LayerNorm(normalized_shape=config['dim_model'])
        self.dropout = nn.Dropout(p=config['hidden_dropout_prob'])
        
        truncated_normal_(self.segments_embedding_layer.weight)
        self.segments_embedding_layer.weight.data[2].zero_()
        truncated_normal_(self.positions_embedding)
        # self.positions_embedding_layer.data[128].zero_()
        self.layer_norm.weight.data.fill_(1.0)
        self.layer_norm.bias.data.zero_()
        
    def forward(self, batched_tokens, batched_segments):
        tokens_embedding = self.tokens_embedding_layer(batched_tokens)#N, L ,D
        segments_embedding = self.segments_embedding_layer(batched_segments)#N, L ,D
        # positions_embedding = self.positions_embedding_layer(batched_positions)#N, L ,D
        
        embedding = tokens_embedding + segments_embedding + self.positions_embedding
        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)
        return embedding #N, L ,D

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(FeedForwardNetwork, self).__init__()
        self.innerlayer = nn.Linear(in_features=config["dim_model"], out_features=config["dim_ff"]) 
        self.activelayer = nn.GELU()
        self.outerlayer = nn.Linear(in_features=config["dim_ff"], out_features=config["dim_model"])
        
        truncated_normal_(self.innerlayer.weight)
        self.innerlayer.bias.data.zero_()
        truncated_normal_(self.outerlayer.weight)
        self.outerlayer.bias.data.zero_()
    
    def forward(self, input):
        """
        * shape *
        input: N, L ,D
        """
        intermediate = self.activelayer(self.innerlayer(input))
        output = self.outerlayer(intermediate)
        return output

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
        multi_outputs = torch.matmul(attn_score, value).transpose(1,2).reshape(VN, QL, self.d_model)  #N, H, QL, L * N, H, L, D/H => N, H, QL, D/H => N, QL, H, D/H => N, QL, D
        output = self.WO(multi_outputs) #N, QL, d_m
        return output
    
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.multihead_layer = MultiHeadAttention(config)
        self.dropout_0 = nn.Dropout(p=config['hidden_dropout_prob'])
        self.layer_norm_0 = nn.LayerNorm(normalized_shape=config['dim_model'])
        self.ff_layer = FeedForwardNetwork(config)
        self.dropout_1 = nn.Dropout(p=config['hidden_dropout_prob'])
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=config['dim_model'])
        
        self.layer_norm_0.weight.data.fill_(1.0)
        self.layer_norm_0.bias.data.zero_()
        self.layer_norm_1.weight.data.fill_(1.0)
        self.layer_norm_1.bias.data.zero_()
    
    def forward(self, input, mask_info):
        mha_output, _ = self.multihead_layer(Query=input, Key=input, Value=input, masked_info=mask_info)
        ff_intput = self.layer_norm_0(self.dropout_0(mha_output)+input)
        ff_output = self.ff_layer(ff_intput)
        layer_output = self.layer_norm_1(self.dropout_1(ff_output)+ff_intput)
        return layer_output
        