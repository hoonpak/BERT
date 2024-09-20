import torch
from torch import nn

from util import truncated_normal_
from sublayer import MultiHeadAttention

class EmbeddingLayer(nn.Module):
    def __init__(self, config, sharing_embedding):
        super(EmbeddingLayer, self).__init__()
        self.tokens_embedding_layer = sharing_embedding
        self.segments_embedding_layer = nn.Embedding(num_embeddings=config['types'], embedding_dim=config['dim_model'], padding_idx=2)
        # self.positions_embedding_layer = nn.Embedding(num_embeddings=config['max_position_embeddings']+1, embedding_dim=config['dim_model'], padding_idx=128)
        self.positions_embedding = nn.Parameter(torch.rand(config['max_position_embeddings']+1, config['dim_model']))
        
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
        mha_output = self.multihead_layer(Query=input, Key=input, Value=input, masked_info=mask_info)
        ff_intput = self.layer_norm_0(self.dropout_0(mha_output)+input)
        ff_output = self.ff_layer(ff_intput)
        layer_output = self.layer_norm_1(self.dropout_1(ff_output)+ff_intput)
        return layer_output
        