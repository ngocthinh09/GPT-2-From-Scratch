"""
FineWeb-Edu dataset for educational content, including textbooks, lecture notes, and academic papers.
Link: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as: `python data/fineweb-edu.py` will download the data, tokenize it, and save the tokenized data to the local directory 'data/fineweb-edu-10B/'
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Configuration
local_dir = 'fineweb-edu-10B'
remote_name = 'sample-10BT'
shard_size = int(1e8)
max_shards = 3
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)

os.makedirs(DATA_CACHE_DIR, exist_ok=True)


# Load the dataset
fwedu = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", cache_dir=DATA_CACHE_DIR, streaming=True)
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def write_datafile(filename, data):
    """Write a shard of tokenized data to disk."""
    np.save(filename, data)
    
shard_index = 0
all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
token_count = 0
progress_bar = None
finished = False


print(f'Starting streaming and processing data to {local_dir}')

for doc in fwedu:
    if finished: break
    
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc['text']))
    tokens = np.array(tokens, dtype=np.uint16)
    
    token_ptr = 0
    while token_ptr < len(tokens):
        space_left = shard_size - token_count
        chunk = tokens[token_ptr : token_ptr + space_left]
        all_tokens_np[token_count : token_count + len(chunk)] = chunk
        token_ptr += len(chunk)
        token_count += len(chunk)
        
        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(chunk))
        
        if token_count == shard_size:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np)
            
            shard_index += 1
            progress_bar = None
            token_count = 0
            
            if max_shards != -1 and shard_index >= max_shards:
                print(f"Reached max shards limit of {max_shards}. Stopping.")
                if progress_bar is not None:
                    progress_bar.close()
                os._exit(0)