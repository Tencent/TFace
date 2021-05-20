from torch.nn import CrossEntropyLoss
from torchkit.loss.dist_softmax import DistCrossEntropy
from torchkit.loss.focal import FocalLoss
from torchkit.loss.ddl import DDL

_loss_dict = {
    'Softmax': CrossEntropyLoss(),
    'DistCrossEntropy': DistCrossEntropy(),
    'FocalLoss': FocalLoss(),
    'DDL': DDL()
}


def get_loss(key):
    """ Get different training loss functions by key,
        support Softmax(distfc = False), DistCrossEntropy (distfc = True), FocalLoss, and DDL.
    """
    if key in _loss_dict.keys():
        return _loss_dict[key]
    else:
        raise KeyError("not support loss {}".format(key))
