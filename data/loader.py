import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import numpy as np
import os
import logging

# Configuration
data_folder = 'fineweb-edu-10B'

def load_tokens(filename):
    """Load a shard of tokenized data from disk."""
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B: int, T: int, process_rank, num_processes, split: str = 'train'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.logger = logging.getLogger(f'NanoGPT.{__name__}')
        
        assert (split in ['train', 'val']), "split must be 'train' or 'val'"
        data_root = os.path.join(os.path.dirname(__file__), data_folder)
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards.sort()
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert (len(self.shards) > 0), f"No shards found for split '{split}' in {data_root}"
        if process_rank == 0:
            self.logger.info(f"DataLoaderLite found {len(self.shards)} shards for split '{split}' in {data_root}")
        self.reset()
        
    def reset(self):
        self.current_shard_index = 0
        self.tokens = load_tokens(self.shards[self.current_shard_index])
        self.current_position = self.B * self.T * self.process_rank
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.current_shard_index = (self.current_shard_index + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard_index])
            self.current_position = self.B * self.T * self.process_rank
        return x, y