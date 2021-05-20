import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='convert training index file')
    parser.add_argument('--old', default=None, type=str, required=True,
                        help='path to old training list')
    parser.add_argument('--tfr_index', default=None, type=str, required=True,
                        help='path to tfrecord index file')
    parser.add_argument('--new', default=None, type=str, required=True,
                        help='path to new training list')
    args = parser.parse_args()
    return args


def build_dict(tfr_index):
    d = {}
    print("reading {}".format(tfr_index))
    tfr_name = os.path.basename(tfr_index).replace('.index', '')
    with open(tfr_index, 'r') as f:
        for line in f:
            file_name, shard_index, offset = line.rstrip().split('\t')
            d[file_name] = '{}\t{}\t{}'.format(tfr_name, shard_index, offset)
    print("build dict done")
    return d


def convert(index_file, d, out_index_file):
    print("write to new index file {}".format(out_index_file))
    with open(index_file, 'r') as f, open(out_index_file, 'w') as out_f:
        for line in f:
            if '\t' in line:
                file_name, label = line.rstrip().split('\t')
            else:
                file_name, label = line.rstrip().split(' ')
            tfr_string = d.get(file_name, None)
            if tfr_string is None:
                print(file_name + " failed")
                continue
            out_f.write(tfr_string + '\t{}'.format(label) + '\n')


def main():
    args = parse_args()
    d = build_dict(args.tfr_index)
    convert(args.old, d, args.new)


if __name__ == '__main__':
    main()
