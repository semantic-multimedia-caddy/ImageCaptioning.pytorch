import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "ec2-3-35-27-164.ap-northeast-2.compute.amazonaws.com"
    # os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "25555"
    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)    


def cleanup():
    dist.destroy_process_group()

    
""" 변화도 평균 계산하기 """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

    
        