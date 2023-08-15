from .M3DFEL import M3DFEL


def create_model(args):
    """create model according to args

    Args:
        args
    """
    model = M3DFEL(args)

    return model
