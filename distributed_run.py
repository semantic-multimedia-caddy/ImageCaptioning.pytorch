import argparse
import torch
from torch import multiprocessing as mp
import captioning.utils.opts as opts
from distributed_utils import setup, cleanup
from train_distributed import train


def run():
    opt = opts.parse_opt()
    # mp.spawn(train, args=(opt, world_size), nprocs=world_size, join=True)
    train(opt.rank, opt, opt.world_size)
    
    
if __name__ == "__main__":
    run()
    