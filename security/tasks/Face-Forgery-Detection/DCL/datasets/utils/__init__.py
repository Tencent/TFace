"""
This module contains the following main classes/functions:
    - data_structure (function):
        helper function to set up FaceForenisics++ dataset
    - SRM (function):
        SRM related weights and convolutional layers
    - RandomPatch (class):
        random patch data augmentation
"""
from .data_structure import *
from .srm import setup_srm_weights, setup_srm_layer
from .random_patch import RandomPatch
