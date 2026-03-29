import os
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup():
    ddp = int(os.environ.get('RANK', -1)) != -1
    
    if ddp:
        assert torch.cuda.is_available(), "Distributed training requires CUDA"
        init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
        master_process = (rank == 0)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        master_process = True
        device = 'cpu'
        if (torch.cuda.is_available()):
            device = f'cuda:{local_rank}'
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = 'mps'
    
    return ddp, rank, local_rank, world_size, master_process, device

def ddp_cleanup():
    if dist.is_initialized():
        destroy_process_group()
        