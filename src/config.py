from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', use_fast=True)
pad_idx = tokenizer.convert_tokens_to_ids("[PAD]")

small_config = {"dim_model": 512,
                "hidden_act": "gelu",
                "init_range": 0.02,
                "vocab_size": 30522,
                "hidden_dropout_prob": 0.1, 
                "num_attention_heads": 8, 
                "types": 3, 
                "max_position_embeddings": 128, 
                "num_hidden_layers": 4, 
                "dim_ff": 2048, 
                "attention_probs_dropout_prob": 0.1,
                "pad_idx": pad_idx}

tiny_config = {"dim_model": 128,
               "hidden_act": "gelu",
               "init_range": 0.02,
               "vocab_size": 30522,
               "hidden_dropout_prob": 0.1,
               "num_attention_heads": 2,
               "types": 3,
               "max_position_embeddings": 128,
               "num_hidden_layers": 2,
               "dim_ff": 512,
               "attention_probs_dropout_prob": 0.1,
               "pad_idx": pad_idx}