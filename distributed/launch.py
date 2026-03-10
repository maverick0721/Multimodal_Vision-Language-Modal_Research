import torch.multiprocessing as mp

def launch(fn,world_size):

    mp.spawn(
        fn,
        args=(world_size,),
        nprocs=world_size
    )