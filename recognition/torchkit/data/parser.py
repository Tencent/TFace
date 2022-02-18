import cv2
import numpy as np
import dareblopy as db
from . import example_pb2


class IndexParser(object):
    """ Class for Line parser.
    """
    def __init__(self) -> None:
        self.sample_num = 0
        self.class_num = 0

    def __call__(self, line):
        line_s = line.rstrip().split('\t')
        if len(line_s) == 2:
            # Default line format
            img_path, label = line_s
            label = int(label)
            self.sample_num += 1
            self.class_num = max(self.class_num, label)
            return (img_path, label)
        elif len(line_s) == 4:
            # IndexTFRDataset line format
            tfr_name, tfr_index, tfr_offset, label = line_s
            label = int(label)
            tfr_file = "{0}/{0}-{1:05d}.tfrecord".format(tfr_name, int(tfr_index))
            tfr_offset = int(tfr_offset)
            self.sample_num += 1
            self.class_num = max(self.class_num, label)
            return (tfr_file, tfr_offset, label)
        else:
            raise RuntimeError("IndexParser line length %d not supported" % len(line_s))

    def reset(self):
        self.sample_num = 0
        self.class_num = 0


class ImgSampleParser(object):
    """ Class for Image Sample parser
    """
    def __init__(self, transform) -> None:
        self.transform = transform

    def __call__(self, path, label):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class TFRecordSampleParser(object):
    """ Class for TFRecord Sample parser
    """
    def __init__(self, transform) -> None:
        self.transform = transform
        self.file_readers = dict()

    def __call__(self, record_path, offset, label):
        rr = self.file_readers.get(record_path, None)
        if rr is None:
            rr = db.RecordReader(record_path)
            self.file_readers[record_path] = rr
        pb_data = rr.read_record(offset)
        example = example_pb2.Example()
        example.ParseFromString(pb_data)
        image_raw = example.features.feature['image'].bytes_list.value[0]
        image = cv2.imdecode(np.frombuffer(image_raw, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
