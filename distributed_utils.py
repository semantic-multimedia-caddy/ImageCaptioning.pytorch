import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(opt):
    os.environ["MASTER_ADDR"] = opt.master
    # os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{opt.port}"
    os.environ["RANK"] = f"{opt.rank}"
    os.environ["WORLD_SIZE"] = f"{opt.world_size}"
    
    print("Connecting...")
    dist.init_process_group("nccl", rank=opt.rank, world_size=opt.world_size)    
    print("Connected!")


def cleanup():
    dist.destroy_process_group()

    
""" 변화도 평균 계산하기 """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

    
        