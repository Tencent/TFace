def get_head(key, dist_fc):
    """ Get different classification head functions by key, support NormFace CosFace, ArcFace, CurricularFace.
        If distfc is True, the weight is splited equally into all gpus and calculated in parallel
    """
    if dist_fc:
        from torchkit.head.distfc.arcface import ArcFace
        from torchkit.head.distfc.cosface import CosFace
        from torchkit.head.distfc.curricularface import CurricularFace
        from torchkit.head.distfc.normface import NormFace
        _head_dict = {
            'CosFace': CosFace,
            'ArcFace': ArcFace,
            'CurricularFace': CurricularFace,
            'NormFace': NormFace
        }
    else:
        from torchkit.head.localfc.cosface import CosFace
        from torchkit.head.localfc.arcface import ArcFace
        from torchkit.head.localfc.curricularface import CurricularFace
        from torchkit.head.localfc.cifp import Cifp
        _head_dict = {
            'CosFace': CosFace,
            'ArcFace': ArcFace,
            'CurricularFace': CurricularFace,
            'Cifp': Cifp,
        }
    if key in _head_dict.keys():
        return _head_dict[key]
    else:
        raise KeyError("not support head {}".format(key))
