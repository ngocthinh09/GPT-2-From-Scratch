import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

class DataLoaderLite:
    def __init__(self, B: int, T: int, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        with open('input.txt', 'r') as file:
            text = file.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        
        if (process_rank == 0):
            print(f'DataLoaderLite initialized with {len(self.tokens)} tokens.')
            print('Epoch size (number of batches per epoch):', len(self.tokens) // (B * T))
            print(f'Batch size: {self.B}, Sequence length: {self.T}')
        
        self.current_position = self.B * self.T * self.process_rank
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y