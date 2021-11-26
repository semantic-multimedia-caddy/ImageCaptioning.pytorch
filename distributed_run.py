import argparse
import torch
from torch import multiprocessing as mp
import captioning.utils.opts as opts
from distributed_utils import setup, cleanup
from train_distributed import train


def run():
    opt = opts.parse_opt()
    # mp.spawn(train, args=(opt, opt.world_size), nprocs=1, join=True)
    train(opt)
    
    
if __name__ == "__main__":
    run()
    