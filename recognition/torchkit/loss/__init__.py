from torch.nn import CrossEntropyLoss
from .dist_softmax import DistCrossEntropy
from .ddl import DDL

_loss_dict = {
    'Softmax': CrossEntropyLoss(),
    'DistCrossEntropy': DistCrossEntropy(),
    'DDL': DDL()
}


def get_loss(key):
    """ Get different training loss functions by key,
        support Softmax(distfc = False), DistCrossEntropy (distfc = True), and DDL.
    """
    if key in _loss_dict.keys():
        return _loss_dict[key]
    else:
        raise KeyError("not support loss {}".format(key))
