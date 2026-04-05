from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024              # max sequence length
    vocab_size: int = 50257             # number of tokens: 50,000 BPE tokens + 256 bytes token + 1 <|endoftext|> token
    n_layer: int = 12                   # number of layers
    n_head: int = 12                    # number of heads
    n_embd: int = 768                   # embedding dimensions
    
GPT2_CONFIGS = {
    "gpt2":         dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
    "gpt2-medium":  dict(n_layer=24, n_head=16, n_embd=1024), # 345M params
    "gpt2-large":   dict(n_layer=36, n_head=20, n_embd=1280), # 762M params
    "gpt2-xl":      dict(n_layer=48, n_head=25, n_embd=1600), # 1542M params
}

def get_model_config(model_type: str, **kwargs) -> GPTConfig:
    if model_type not in GPT2_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(GPT2_CONFIGS.keys())}")
    config_params = GPT2_CONFIGS[model_type].copy()
    config_params.update(kwargs)
    return GPTConfig(**config_params)

@dataclass
class LRSchedulerConfig:
    max_lr: float = 6e-4                # peak learning rate
    min_lr: float = 6e-5                # minimum learning rate (after decay)
    weigh_decay: float = 0.1            # weight decay for AdamW optimizer

@dataclass
class TrainingConfig:
    total_batch_size: int = 524288      # total batch size across all devices and gradient accumulation steps
    B: int = 8                         # micro-batch size per device
    T: int = 1024                       # sequence length
    warmup_steps: int = 21              # number of steps to warm up the learning rate
    max_steps: int = 381                # total number of training steps

@dataclass
class SamplingConfig:
    temperature: float = 0.7            # sampling temperature
    num_return_sequences: int = 4       # number of sequences to generate
    max_length: int = 32                # maximum length of generated sequences
    prompt: str = "Once upon a time"    # initial prompt for generation
    