import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed import ReduceOp


class AllGatherFunc(Function):
    """ AllGather op with gradient backward
    """
    @staticmethod
    def forward(ctx, tensor, world_size):
        gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gather_list, tensor)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        grad_list = list(grads)
        grad_out = torch.zeros_like(grad_list[rank], requires_grad=True)
        dist.reduce_scatter(grad_out, grad_list, op=ReduceOp.SUM)
        # Gradient correction for DistCrossEntropy
        grad_out = grad_out * world_size
        return (grad_out, None)


AllGather = AllGatherFunc.apply
