import os
import struct
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from torchkit.data import example_pb2


def read_index_file(index_file):
    """ Parse index file, each line contains record_name, record_index, record_offset and label
    """
    samples_offsets = []
    record_files = []
    labels = []
    with open(index_file, 'r') as ifs:
        for line in ifs:
            record_name, tf_record_index, tf_record_offset, label = line.rstrip().split('\t')
            samples_offsets.append(int(tf_record_offset))
            record_files.append(os.path.join(
                record_name,
                "%s-%05d.tfrecord" % (record_name, int(tf_record_index))
            ))
            labels.append(int(label))
    return record_files, samples_offsets, labels


def read_pair_index_file(index_file):
    """ Parse pair index file, each line contains a pair of (record_name, record_index, record_offset)
    """
    samples_offsets = []
    record_files = []
    labels = []
    with open(index_file, 'r') as ifs:
        for line in ifs:
            (record_name_first, tf_record_index_first, tf_record_offset_first,
             label_first, record_name_second, tf_record_index_second,
             tf_record_offset_second, label_second) = line.rstrip().split('\t')
            samples_offsets.append((int(tf_record_offset_first), int(tf_record_offset_second)))
            record_files.append((os.path.join(
                record_name_first,
                "%s-%05d.tfrecord" % (record_name_first, int(tf_record_index_first))
            ), os.path.join(
                record_name_second,
                "%s-%05d.tfrecord" % (record_name_second, int(tf_record_index_second)))))
            labels.append((int(label_first), (label_second)))

    return (record_files, samples_offsets, labels)


class IndexTFRDataset(Dataset):
    """ Index TFRecord Dataset
    """

    def __init__(self, tfrecord_dir, index_file, transform):
        """ Create a ``IndexTFRDataset`` object
            A ``IndexTFRDataset`` object will read sample proto from *.tfrecord files saved
            in ``tfrecord_dir`` by index_file, the sample proto will convert to image and
            fed into Dataloader.

            Args:
                tfrecord_dir: tfrecord saved dir
                index_file: each line format ``tfr_name\t tfr_record_index \t tfr_record_offset \t label``
                transform: image transform
        """
        self.root_dir = tfrecord_dir
        self.index_file = index_file
        self.transform = transform
        self.records, self.offsets, self.labels = read_index_file(self.index_file)
        for record_file in set(self.records):
            record_file = os.path.join(self.root_dir, record_file)
            if not os.path.exists(record_file):
                raise RuntimeError("tfrecord file： %s not found" % record_file)
        self.class_num = max(self.labels) + 1
        self.sample_num = len(self.records)
        print('class_num: %d, sample_num:  %d' % (self.class_num, self.sample_num))

    def __len__(self):
        return self.sample_num

    def _parser(self, feature_list):
        for key, feature in feature_list:
            if key == 'image':
                image_raw = feature.bytes_list.value[0]
                image = Image.open(BytesIO(image_raw))
                image = image.convert('RGB')
                image = self.transform(image)
        return image

    def _get_record(self, record_file, offset):
        with open(record_file, 'rb') as ifs:
            ifs.seek(offset)
            byte_len_crc = ifs.read(12)
            proto_len = struct.unpack('Q', byte_len_crc[:8])[0]
            pb_data = ifs.read(proto_len)
            if len(pb_data) < proto_len:
                print("read pb_data err,proto_len:%s pb_data len:%s" % (proto_len, len(pb_data)))
                return None
        example = example_pb2.Example()
        example.ParseFromString(pb_data)
        # keep key value in order
        feature = sorted(example.features.feature.items())
        record = self._parser(feature)
        return record

    def __getitem__(self, index):
        offset = self.offsets[index]
        record = self.records[index]
        record_file = os.path.join(self.root_dir, record)
        return self._get_record(record_file, offset), self.labels[index]


class PairIndexTFRDataset(IndexTFRDataset):
    """ PairIndex TFRecord Dataset
    """

    def __init__(self, tfrecord_dir, index_file, transform):
        """ Create a ``PairIndexTFRDataset`` object
            A ``PairIndexTFRDataset`` object will read sample proto from all related *.tfrecord files saved
            in ``tfrecord_dir`` by pair_index_file, the sample proto will convert to image and
            fed into Dataloader.

            Args:
                tfrecord_dir: tfrecord saved dir
                index_file: each line format ``tfr_name\t tfr_record_index \t tfr_record_offset \t label
                                            tfr_name\t tfr_record_index \t tfr_record_offset \t label``
                transform: image transform
        """
        self.root_dir = tfrecord_dir
        self.index_file = index_file
        self.transform = transform
        self.records, self.offsets, self.labels = read_pair_index_file(self.index_file)
        records_first, records_second = map(list, zip(*self.records))
        for record_file in set(records_first):
            record_file = os.path.join(self.root_dir, record_file)
            if not os.path.exists(record_file):
                raise RuntimeError("tfrecord file： %s not found" % record_file)
        for record_file in set(records_second):
            record_file = os.path.join(self.root_dir, record_file)
            if not os.path.exists(record_file):
                raise RuntimeError("tfrecord file： %s not found" % record_file)

        self.sample_num = len(self.records)
        first_labels, _ = map(list, zip(*self.labels))
        self.class_num = max(first_labels) + 1
        print('class_num: %d, sample_num:  %d' % (self.class_num, self.sample_num))

    def __getitem__(self, index):
        first_offset, second_offset = self.offsets[index]
        first_record, second_record = self.records[index]
        first_record_file = os.path.join(self.root_dir, first_record)
        second_record_file = os.path.join(self.root_dir, second_record)
        first_image = self._get_record(first_record_file, first_offset)
        second_image = self._get_record(second_record_file, second_offset)

        return first_image, second_image, self.labels[index][0]
