import torch.distributed as dist

def setup():

    dist.init_process_group(
        backend="nccl"
    )

def cleanup():

    dist.destroy_process_group()