import argparse
import torch
import time
from torch import multiprocessing as mp
import captioning.utils.opts as opts
from distributed_utils import setup, cleanup
from train_distributed2 import train


def run():
    opt = opts.parse_opt()
    # mp.spawn(train, args=(opt, opt.world_size), nprocs=1, join=True)

    now = time.time()
    train(opt)
    duration = time.time() - now

    print(f"Duration: {duration} sec(s).")
    
    
if __name__ == "__main__":
    run()
    