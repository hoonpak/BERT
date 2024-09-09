import torch
from torch import nn
from layer import EmbeddingLayer, EncoderLayer

from util import truncated_normal_

class BERT(nn.Module):
    def __init__(self, config, sharing_embedding):
        super(BERT, self).__init__()
        self.embedding_layer = EmbeddingLayer(config=config, sharing_embedding=sharing_embedding)
        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config['num_hidden_layers'])])
        self.pooling_layer = nn.Linear(in_features=config['dim_model'], out_features=config['dim_model'])
        self.tanh = nn.Tanh()
        self.padding_idx = config['pad_idx']
        
    def forward(self, batched_tokens, batched_segments, batched_positions):
        mask_info = (batched_tokens == self.padding_idx) # padding got True value, so it ignore during computing attention.
        x = self.embedding_layer(batched_tokens, batched_segments, batched_positions)
        for enc_layer in self.encoder_layers:
            x = enc_layer.forward(input=x, mask_info=mask_info)
        pooled_output = self.tanh(self.pooling_layer(x[:,0,:]))
        return x, pooled_output
    
class NextSentencePrediction(nn.Module):
    def __init__(self, config):
        super(NextSentencePrediction, self).__init__()
        self.cls_project_layer = nn.Linear(in_features=config['dim_model'], out_features=2)
        
        truncated_normal_(self.cls_project_layer.weight, mean=0.0, std=config['init_range'])
        nn.init.zeros_(self.cls_project_layer.bias)
        
    def forward(self, input):
        """
        ** shape **
        input - N, 1, H => these are hidden values which was position cls tokens..
        output - N, 1, 2
        """
        nsp_predict = self.cls_project_layer(input)
        return nsp_predict # -> input of the CE
        
class MaskedLanguageModeling(nn.Module):
    def __init__(self, config, sharing_embedding):
        super(MaskedLanguageModeling, self).__init__()
        self.dense_layer = nn.Linear(in_features=config['dim_model'], out_features=config['dim_model'])
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(normalized_shape=config['dim_model'])
        self.cls_layer = nn.Linear(in_features=config['dim_model'], out_features=config['vocab_size']) #it is sharing the parameters with token embedding.
        
        truncated_normal_(self.dense_layer.weight, mean=0.0, std=config['init_range'])
        nn.init.zeros_(self.dense_layer.bias)
        truncated_normal_(self.layer_norm.weight, mean=0.0, std=config['init_range'])
        nn.init.zeros_(self.layer_norm.bias)
        self.cls_layer.weight = sharing_embedding.weight
        nn.init.zeros_(self.cls_layer.bias)
        
        self.vocab_size = config['vocab_size']
        
    def forward(self, input):
        masked_output = self.gelu(self.dense_layer(input))
        norm_masked_output = self.layer_norm(masked_output)
        cls_output = self.cls_layer(norm_masked_output).reshape(-1, self.vocab_size)
        return cls_output
        