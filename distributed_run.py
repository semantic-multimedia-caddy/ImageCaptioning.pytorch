import torch
from torch import multiprocessing as mp
import captioning.utils.opts as opts
from distributed_utils import setup, cleanup


NUM_WORKERS = 2


def run():
    opt = opts.parse_opt()
    world_size = NUM_WORKERS
    mp.spawn(train, args=(opt, world_size), nprocs=world_size, join=True)
    
    
if __name__ == "__main__":
    run()
    