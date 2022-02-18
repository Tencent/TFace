import argparse
from omegaconf import OmegaConf


def flatten_dict(dictionary, delimiter ='.'):
    flat_dict = dict()
    for key, value in dictionary.items():
        if isinstance(value, dict):
            flatten_value_dict = flatten_dict(value, delimiter)
            for k, v in flatten_value_dict.items():
                flat_dict[f"{key}{delimiter}{k}"] = v
        else:
            flat_dict[key] = value
    return flat_dict


def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def warn_print(x):
    """Print info in the highlighted effect.

    Args:
        x (any): Info to print.
    """
    x = str(x)
    x = "\x1b[33;1m" + x + "\x1b[0m"
    print(x)


def get_params(to_dict=False, **new_kwargs):
    """Parse the parameters from a yaml config file, and allow for param modification in the command line.
    The input '-c' or '--config' must be specified.
    Three arguments have default presets: distributed, local_rank and world_size.
    
    Argument priority: cmd args > new_kwargs > dict in config.

    Args:
        to_dict (bool, optional): 
            Whether to return the parsed args in Python Dict type. Defaults to False.
        new_kwargs (**kwargs):
            Allow for overloading some params by specifying new params here.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config', type=str, default='cfg.yaml')
    parser.add_argument('--distributed', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)

    # parse the above cmd options
    args_tmp = parser.parse_known_args()[0]
    args_tmp_dict = vars(args_tmp)

    oc_cfg = OmegaConf.load(args_tmp.config)
    # args_tmp_dict.pop('config')
    oc_cfg.merge_with(args_tmp_dict)

    # append items from new_kwargs
    if new_kwargs:
        for k in new_kwargs:
            if k in oc_cfg:
                warn_print(f'{k} from `new_kwargs` found in original conf, will keep the one in `new_kwargs`')
        oc_cfg.merge_with(OmegaConf.create(new_kwargs))

    oc_cfg_dict = OmegaConf.to_container(oc_cfg, resolve=True)

    oc_cfg_dict_flatten = flatten_dict(oc_cfg_dict)

    # add options from config.yaml to argparse
    for k, v in oc_cfg_dict_flatten.items():
        if k in args_tmp_dict:
            continue
        if isinstance(v, bool):
            parser.add_argument('--{}'.format(k), dest=k.replace('.', '___'), 
                type=lambda x: (str(x).lower() == 'true'), default=v)
        elif isinstance(v, list) or isinstance(v, tuple):
            parser.add_argument('--{}'.format(k), dest=k.replace('.', '___'), 
                type=type(v[0]), default=v, nargs='+')
        else:
            parser.add_argument('--{}'.format(k), dest=k.replace('.', '___'), 
                type=str if v is None else type(v), default=v)
    parser.add_argument('-h', '--help', action='help', help=('show this help message and exit'))
    args = parser.parse_args()

    var_args = vars(args)
    for k, v in var_args.items():
        # if k == 'config':
        #     continue
        ori_k = k.replace('___', '.')
        sub_ks = ori_k.split('.')
        nested_set(oc_cfg_dict, sub_ks, v)

    oc_cfg.merge_with(oc_cfg_dict)

    # print(OmegaConf.to_yaml(oc_cfg))
    if to_dict:
        return OmegaConf.to_container(oc_cfg, resolve=True)
    return oc_cfg
