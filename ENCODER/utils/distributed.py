import torch
import torch.distributed as dist

@torch.no_grad()
def concat_all_gather(tensor):
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    tensors_gather = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)

@torch.no_grad()
def batch_shuffle_ddp(x):
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]
    if batch_size_all % batch_size_this != 0:
        raise RuntimeError("Mismatch crítico en multi-GPU durante shuffle.")
    num_gpus = batch_size_all // batch_size_this
    idx_shuffle = torch.randperm(batch_size_all, device=x.device)
    dist.broadcast(idx_shuffle, src=0)
    idx_this = idx_shuffle.view(num_gpus, -1)[dist.get_rank()]
    return x_gather[idx_this], idx_shuffle

@torch.no_grad()
def batch_unshuffle_ddp(x, idx_shuffle):
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]
    if batch_size_all % batch_size_this != 0:
        raise RuntimeError("Mismatch crítico en multi-GPU durante unshuffle.")
    num_gpus = batch_size_all // batch_size_this
    idx_unshuffle = torch.argsort(idx_shuffle)
    idx_this = idx_unshuffle.view(num_gpus, -1)[dist.get_rank()]
    return x_gather[idx_this]