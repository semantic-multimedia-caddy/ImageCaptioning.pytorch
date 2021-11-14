import torch
from torch import multiprocessing as mp
import captioning.utils.opts as opts
from distributed_utils import setup, cleanup
from train import train


NUM_WORKERS = 2
RANK = 0


def run():
    opt = opts.parse_opt()
    world_size = NUM_WORKERS
    # mp.spawn(train, args=(opt, world_size), nprocs=world_size, join=True)
    train(RANK, opt, NUM_WORKERS)
    
    
if __name__ == "__main__":
    run()
    