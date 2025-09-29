import os
import sys
import argparse
import numpy as np
import torch
from utils import perform_val_bin
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from torchkit.backbone import get_model
from utils import perform_rfw_val_bin


def parse_args():
    parser = argparse.ArgumentParser(description='verfication tool')
    parser.add_argument('--ckpt_path', default=None, required=True, help='model_path')
    parser.add_argument('--backbone', default='IR_34', help='backbone type')
    parser.add_argument('--gpu_ids', default='0', help='gpu ids')
    parser.add_argument('--batch_size', default=64, help='batch size')
    parser.add_argument('--data_root', default='', required=True, help='validation data root')
    parser.add_argument('--embedding_size', default=512, help='embedding_size')
    args = parser.parse_args()
    return args


def load_rfw_bin(path, name):
    """ read data for bin files
        """
    import pickle
    import cv2

    path = os.path.join(path, name)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data_list = [[], []]
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        img = np.frombuffer(_bin, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, axes=(2, 0, 1))
        img = (img / 255. - 0.5) / 0.5
        img = img.astype(np.float32)
        data_list[0].append(img)
        data_list[1].append(np.flip(img, axis=2))
    data_list[0] = np.array(data_list[0], dtype=np.float32)
    data_list[1] = np.array(data_list[1], dtype=np.float32)
    return data_list, issame_list


def main():
    """ Perform evaluation on RFW datasets,
        each dataset consists of some positive and negative pair data.
    """
    args = parse_args()
    torch.manual_seed(1337)
    input_size = [112, 112]
    val_data_dir = args.data_root
    batch_size = args.batch_size

    # load backbone
    backbone = get_model(args.backbone)(input_size)
    if not os.path.exists(args.ckpt_path):
        raise RuntimeError("%s not exists" % args.ckpt_path)
    backbone.load_state_dict(torch.load(args.ckpt_path, weights_only=True))
    # backbone to gpu
    gpus = [int(x) for x in args.gpu_ids.rstrip().split(',')]
    if len(gpus) > 1:
        backbone = torch.nn.DataParallel(backbone, device_ids=gpus)
    backbone = backbone.cuda()

    accuracies = []
    race_bins = ['African_test.bin', 'Asian_test.bin', 'Caucasian_test.bin', 'Indian_test.bin']
    for race_bin in race_bins:
        # perform accuracy evaluation on each ethnicity
        ver_list = load_rfw_bin(val_data_dir, race_bin)
        results = []
        acc1, std1, acc2, std2, xnorm, embeddings_list = perform_rfw_val_bin(ver_list, backbone, gpus, batch_size)
        print('[%s]XNorm: %f' % (race_bin, xnorm))
        print('[%s]Accuracy: %1.5f+-%1.5f' % (race_bin, acc1, std1))
        print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (race_bin, acc2, std2))
        results.append(acc2)
        print('Max of [%s] is %1.5f' % (race_bin, np.max(results)))
        accuracies.append(np.max(results))
    accuracies = np.array(accuracies)
    # calculate accuracy mean and standard
    print('Mean of accurary: {}, Std of accuracy: {}'.format(np.mean(accuracies), np.std(100 * accuracies, ddof=1)))


if __name__ == "__main__":
    main()
