from dataclasses import dataclass
from .constants import *
from easydict import EasyDict
from dataclasses import dataclass

def get_additional_params(config):
    # TODO generalize it
    # consider for now that nn_hop_length == window_length
    config.nn_hop_length = config.window_length
    n_samples_in_nn_hop = int(config.sr * config.nn_hop_length)
    n_samples_in_window = int(config.sr * config.window_length)    
    config.n_samples_in_nn_hop = n_samples_in_nn_hop
    config.n_samples_in_window = n_samples_in_window
    
    return config
