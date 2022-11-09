xprint = print

from .datapool import DataPool
from .video import Video
from .utils import *
from .loaders import *
from .dtw import *
from .visualization import *
from .constants import *
from .config import *
from .inference import *
from .transformation import *
from .dataset import *
from .model import *
from .structured_predictor_utils import *
from .utils_svm import *
from .metric import *
from .fault_detection import *

try:
    from . import structured_predictor
except ModuleNotFoundError:
    print("Structured predictor is not available")


import os
import sys
import yaml
import hydra
import pickle
import shutil
import random
import pandas as pd

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from tqdm.rich import tqdm_rich
from rich import print
from glob import glob
from omegaconf import DictConfig, OmegaConf

import matplotlib as mpl

mpl.rcParams["agg.path.chunksize"] = 10000


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


if is_interactive():
    print = xprint
