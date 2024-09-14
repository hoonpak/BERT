import torch
from torch import nn
from module import BERT, NextSentencePrediction, MaskedLanguageModeling

from util import truncated_normal_

class PretrainingBERT(nn.Module):
    def __init__(self, config):
        super(PretrainingBERT, self).__init__()
        sharing_embedding = nn.Embedding(num_embeddings=config['vocab_size'], embedding_dim=config['dim_model'], padding_idx=config['pad_idx'])
        truncated_normal_(sharing_embedding.weight)
        sharing_embedding.weight.data[config['pad_idx']].zero_()

        self.bert_layer = BERT(config=config, sharing_embedding=sharing_embedding)
        self.nsp_classifier = NextSentencePrediction(config=config)
        self.mlm_classifier = MaskedLanguageModeling(config=config, sharing_embedding=sharing_embedding)

    def forward(self, batched_tokens, batched_segments, mlm_position):
        bert_output, pooled_output = self.bert_layer(batched_tokens, batched_segments)
        N, S = mlm_position.shape
        # print(bert_output.shape)
        # print(mlm_position)
        mlm_input = bert_output[torch.arange(N).unsqueeze(-1), mlm_position]
        nsp_output = self.nsp_classifier(pooled_output)
        mlm_output = self.mlm_classifier(mlm_input)
        return nsp_output, mlm_output

class ClassifierBERT(nn.Module):
    def __init__(self, config, bert_layer, labels_num):
        super(ClassifierBERT, self).__init__()
        self.bert_layer = bert_layer
        self.classifier = nn.Sequential(
            nn.Dropout(p=config['hidden_dropout_prob']),
            nn.Linear(in_features=config['dim_model'], out_features=labels_num)
        )

    def forward(self, batched_tokens, batched_segments):
        bert_output, pooled_output = self.bert_layer(batched_tokens, batched_segments)
        output = self.classifier(pooled_output)
        return output
    
# class GLEUBERT(nn.Module):
    