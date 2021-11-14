import torch
from torch import multiprocessing as mp
import captioning.utils.opts as opts


def run(batch_size, epochs, eta, fine_tune):
    opt = opts.parse_opt()
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(opt, world_size), nprocs=world_size, join=True)
    
    
if __name__ == "__main__":
    run()
    