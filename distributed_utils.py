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
    os.environ["MASTER_ADDR"] = f"{opt.master}"
    # os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{opt.port}"
    os.environ["RANK"] = f"{opt.rank}"
    os.environ["WORLD_SIZE"] = f"{opt.world_size}"
    
    print(f"Connecting... (rank: {opt.rank}, world size: {opt.world_size})")
    dist.init_process_group("nccl", init_method=f"tcp://{opt.master}:{opt.port}", rank=opt.rank, world_size=opt.world_size)    
    print("Connected!")


def cleanup():
    dist.destroy_process_group()

    
""" 변화도 평균 계산하기 """
def average_gradients(model):
    # rank = int(os.environ["RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    size = float(dist.get_world_size())

    for param in model.parameters():
        print("reducing...")

        # if rank == 0:
        #     t = torch.zeros_like(param.grad.data)
        #     for w in range(1, world_size):
        #         dist.recv(t, w)

        #     param.grad.data += t
        # else:
        #     dist.send(param.grad.data, 0)
        dist.reduce(param.grad.data, 0, op=dist.ReduceOp.SUM)
        # param.grad.data = grad_tensor

        print("broadcasting...")
        dist.broadcast(param.grad.data, 0)
        param.grad.data /= size

    
        