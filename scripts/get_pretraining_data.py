import sys
sys.path.append("../src")
from config import small_config

import warnings
warnings.filterwarnings("ignore")

import random
import argparse
import numpy as np
from tqdm import tqdm
from datasets import concatenate_datasets, load_dataset, load_from_disk, Dataset
from transformers import BertTokenizerFast
import json

def truncated_seq_pair(tokens_a, tokens_b, max_num_tokens):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1
        
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def create_masked_lm_prediction(tokens, masked_lm_prob, max_predictions_per_seq, cls_token_id, sep_token_id, msk_token_id):
    cand_indexes = []

    for (i, token) in enumerate(tokens):
        if (token == cls_token_id) or (token == sep_token_id):
            continue
        cand_indexes.append([i])
    random.shuffle(cand_indexes)
    
    output_tokens = list(tokens)
    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens)*masked_lm_prob))))
    
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            
            masked_token = None
            if random.random() < 0.8:
                masked_token = msk_token_id
            else:
                if random.random() < 0.5:
                    masked_token = tokens[index]
                else:
                    masked_token = random.randint(0, small_config["vocab_size"]-1)
            output_tokens[index] = masked_token
            masked_lms.append([index, tokens[index]])
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x[0])
    
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p[0])
        masked_lm_labels.append(p[1])
    return (output_tokens, masked_lm_positions, masked_lm_labels)
    

def generate_processed_sentence(total_dataset, tokenizer, write_file):
    # document shuffling
    num_total_documents = len(total_dataset)
    shuffling_idx = np.arange(num_total_documents).tolist()
    random.shuffle(shuffling_idx)

    cls_token_id = tokenizer.convert_tokens_to_ids("[CLS]")
    sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
    msk_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
    
    for doc_idx in tqdm(shuffling_idx):
        doc = total_dataset[doc_idx]['text']
        doc = doc.split("\n") #document in units of sentence
        
        num_doc_sen = len(doc)
        max_num_tokens = 128 - 3
        target_seq_length = max_num_tokens
        if (random.random() < 0.1) and (num_doc_sen < 10000): #short_seq_prob
            target_seq_length = random.randint(4, max_num_tokens)
        
        i = 0
        current_chunk = []
        current_length = 0
        while i < num_doc_sen:
            encoded_sen = tokenizer.encode(doc[i], add_special_tokens=False)
            if len(encoded_sen) > 0:
                current_chunk.append(encoded_sen)
                current_length += len(encoded_sen)
                
            if (current_length >= target_seq_length) or (i == num_doc_sen-1):
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 3:
                        a_end = random.randint(1,len(current_chunk)-1)
                    
                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])
                    
                    tokens_b = []
                    is_random_next = False
                    if len(current_chunk) == 1 or random.random() < 0.5:
                        is_random_next = True
                        target_b_length = max(2, target_seq_length - len(tokens_a))
                        
                        for _ in range(1000000):
                            random_document_index = random.randint(0, num_total_documents-1)
                            if random_document_index != doc_idx:
                                random_doc = total_dataset[random_document_index]['text']
                                random_doc = random_doc.split("\n")
                                if len(random_doc) > 30:
                                    break
                        
                        random_start = random.randint(0, len(random_doc)-20)
                        for j in range(random_start, len(random_doc)):
                            random_doc_sen_ids = tokenizer.encode(random_doc[j], add_special_tokens=False)
                            tokens_b.extend(random_doc_sen_ids)
                            if len(tokens_b) > target_b_length:
                                break
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])
                    truncated_seq_pair(tokens_a, tokens_b, max_num_tokens)
                    
                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1
                    
                    tokens = []
                    segment_ids = []
                    tokens.append(cls_token_id)
                    segment_ids.append(0)
                    for token in tokens_a:
                        tokens.append(token)
                        segment_ids.append(0)
                    
                    tokens.append(sep_token_id)
                    segment_ids.append(0)
                    
                    for token in tokens_b:
                        tokens.append(token)
                        segment_ids.append(1)
                    tokens.append(sep_token_id)
                    segment_ids.append(1)

                    (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_prediction(
                        tokens, 0.15, 20, cls_token_id, sep_token_id, msk_token_id
                    )

                    pretraining_data = {"tokens": tokens,
                                        "segment_ids": segment_ids,
                                        "is_random_next": is_random_next,
                                        "masked_lm_positions": masked_lm_positions,
                                        "masked_lm_labels": masked_lm_labels
                                        }
                    json.dump(pretraining_data, write_file)
                    write_file.write("\n")
                current_chunk = []
                current_length = 0
            i += 1
        
def main(name):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', use_fast=True)

    bookcorpus = load_from_disk("../dataset/bookcorpus")
    wiki = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
    total_dataset = concatenate_datasets([wiki, bookcorpus])
    
    write_file = open(f'pretraining_{name}.json', 'w')
    generate_processed_sentence(total_dataset, tokenizer, write_file)
    write_file.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    option = parser.parse_args()
    name = option.name
    main(name)