import torch
import torch.distributed as dist


def reduce_tensor(tensor, mean=True):
    """Reduce tensor in the distributed settting.

    Args:
        tensor (torch.tensor): 
            Input torch tensor to reduce.
        mean (bool, optional): 
            Whether to apply mean. Defaults to True.

    Returns:
        [torch.tensor]: Returned reduced torch tensor or.
    """
    rt = tensor.clone()  # The function operates in-place.
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if mean:
        rt /= dist.get_world_size()
    return rt


def gather_tensor(inp, world_size=None, dist_=True, to_numpy=False):
    """Gather tensor in the distributed setting.

    Args:
        inp (torch.tensor): 
            Input torch tensor to gather.
        world_size (int, optional): 
            Dist world size. Defaults to None. If None, world_size = dist.get_world_size().
        dist_ (bool, optional):
            Whether to use all_gather method to gather all the tensors. Defaults to True.
        to_numpy (bool, optional): 
            Whether to return numpy array. Defaults to False.

    Returns:
        (torch.tensor || numpy.ndarray): Returned tensor or numpy array.
    """
    inp = torch.stack(inp)
    if dist_:
        if world_size is None:
            world_size = dist.get_world_size()
        gather_inp = [torch.ones_like(inp) for _ in range(world_size)]
        dist.all_gather(gather_inp, inp)
        gather_inp = torch.cat(gather_inp)
    else:
        gather_inp = inp

    if to_numpy:
        gather_inp = gather_inp.cpu().numpy()

    return gather_inp
