"""
FaceForensics++ dataloader function collections.
"""
import itertools
from enum import auto
from enum import Enum
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Tuple
from typing import Union


class StrEnum(str, Enum):
    """Utility functions

    Args:
        str ([type]) 
        Enum ([type])
    """
    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def argparse(cls, s):
        try:
            return cls[s]
        except KeyError:
            return s


class Compression(StrEnum):
    """Compression types in FaceForesics++ dataset

    Args:
        StrEnum ([type]): [description]
    """
    raw = auto()
    c23 = auto()
    c40 = auto()
    random_compressed = auto()
    masks = auto()


class DataType(StrEnum):
    bounding_boxes = auto()
    face_images = auto()
    face_images_tracked = auto()
    full_images = auto()
    face_information = auto()
    videos = auto()
    resampled_videos = auto()
    images_v1 = auto()
    images_v3 = auto()


class Method:
    REAL_DIR = "original_sequences/"
    FAKE_DIR = "manipulated_sequences/"

    def __init__(self, name: str, is_real: bool):
        """Method class for FaceForensics++ dataset

        Args:
            name (str): 
            is_real (bool): 
        """
        self.name = name
        self.is_real = is_real

    def get_dir_str(self):
        if self.is_real:
            return self.REAL_DIR + self.name
        else:
            return self.FAKE_DIR + self.name

    def __str__(self):
        return self.name


ACTORS = Method("actors", is_real=True)
YOUTUBE = Method("youtube", is_real=True)

DEEP_FAKE_DETECTION = Method("DeepFakeDetection", is_real=False)
DEEPFAKES = Method("Deepfakes", is_real=False)
FACE2FACE = Method("Face2Face", is_real=False)
FACE_SWAP = Method("FaceSwap", is_real=False)
NEURAL_TEXTURES = Method("NeuralTextures", is_real=False)
FACESHIFTER = Method("FaceShifter", is_real=False)


class FaceForensicsDataStructure:

    METHODS = {
        YOUTUBE.name: YOUTUBE,
        DEEPFAKES.name: DEEPFAKES,
        FACE2FACE.name: FACE2FACE,
        FACE_SWAP.name: FACE_SWAP,
        NEURAL_TEXTURES.name: NEURAL_TEXTURES,
        ACTORS.name: ACTORS,
        DEEP_FAKE_DETECTION.name: DEEP_FAKE_DETECTION,
        FACESHIFTER.name: FACESHIFTER,
    }

    ALL_METHODS = list(METHODS.keys())

    MANIPULATED_METHODS = [
        DEEPFAKES.name,
        FACE2FACE.name,
        FACE_SWAP.name,
        NEURAL_TEXTURES.name,
        FACESHIFTER.name,
    ]

    FF_METHODS = [
        YOUTUBE.name,
        DEEPFAKES.name,
        FACE2FACE.name,
        FACE_SWAP.name,
        NEURAL_TEXTURES.name,
        FACESHIFTER.name,
    ]

    def __init__(
        self,
        root_dir: str,
        methods: Iterable[str],
        compressions: Iterable[Union[str, Compression]] = Compression.c23,
        data_types: Iterable[Union[str, DataType]] = DataType.images_v1,
    ):
        """Data Structure of FaceForensics

        Args:
            root_dir (str): dataset root dir
            methods (Iterable[str]): face methods
            compressions (Iterable[Union[str, Compression]], optional):  Defaults to Compression.c23.
            data_types (Iterable[Union[str, DataType]], optional):  Defaults to DataType.images_v1.

        Raises:
            FileNotFoundError: 
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"{self.root_dir} does not exist!")
        self.methods = [self.METHODS[method] for method in methods]
        self.data_types = (data_types if isinstance(data_types, (List, Tuple)) else (data_types, ))
        self.compressions = (compressions if isinstance(compressions, (List, Tuple)) else (compressions, ))

    def get_subdirs(self) -> List[Path]:
        """Returns subdirectories containing datatype """
        return [
            self.root_dir / method.get_dir_str() / str(compression) / str(data_type)
            for method, compression, data_type in itertools.product(self.methods, self.compressions, self.data_types)
        ]
