import random
import numpy as np
from typing import Tuple, Union, Dict, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .distributed import init_distributed
from .logger import LOGGER


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_cuda(opts) -> Tuple[bool, int, torch.device]:
    """
    Initialize CUDA for distributed computing
    """
    if not torch.cuda.is_available():
        assert opts.local_rank == -1, opts.local_rank
        return True, 0, torch.device("cpu")

    # get device settings
    if opts.local_rank != -1:
        init_distributed(opts)
        torch.cuda.set_device(opts.local_rank)
        device = torch.device("cuda", opts.local_rank)
        n_gpu = 1
        default_gpu = dist.get_rank() == 0
        if default_gpu:
            LOGGER.info(f"Found {dist.get_world_size()} GPUs")
    else:
        default_gpu = True
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()

    return default_gpu, n_gpu, device


def wrap_model(
    model: torch.nn.Module, device: torch.device, local_rank: int
) -> torch.nn.Module:
    model.to(device)

    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        # At the time of DDP wrapping, parameters and buffers (i.e., model.state_dict()) 
        # on rank0 are broadcasted to all other ranks.

    return model


class NoOp(object):
    """ useful for distributed training No-Ops """
    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return