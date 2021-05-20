import torch.distributed as dist
from torch.autograd import Function
from torch.distributed import ReduceOp


class AllGatherFunc(Function):
    """ AllGather op with gradient backword
    """
    @staticmethod
    def forward(ctx, tensor, *gather_list):
        gather_list = list(gather_list)
        dist.all_gather(gather_list, tensor)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        grad_list = list(grads)
        rank = dist.get_rank()
        grad_out = grad_list[rank]
        dist_ops = [
            dist.reduce(grad_out, rank, ReduceOp.SUM, async_op=True) if i == rank else
            dist.reduce(grad_list[i], i, ReduceOp.SUM, async_op=True) for i in range(dist.get_world_size())
        ]
        for _op in dist_ops:
            _op.wait()

        grad_out *= len(grad_list)  # cooperate with distributed loss function
        return (grad_out, *[None for _ in range(len(grad_list))])


AllGather = AllGatherFunc.apply
