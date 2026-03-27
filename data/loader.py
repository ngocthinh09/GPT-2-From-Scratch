import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

class DataLoaderLite:
    def __init__(self, B: int, T: int):
        self.B = B
        self.T = T
        
        with open('input.txt', 'r') as file:
            text = file.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        
        print(f'DataLoaderLite initialized with {len(self.tokens)} tokens.')
        print('Epoch size (number of batches per epoch):', len(self.tokens) // (B * T))
        print(f'Batch size: {self.B}, Sequence length: {self.T}')
        
        self.current_position = 0
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + (B*T + 1) >= len(self.tokens):
            self.current_position = 0
        return x, y