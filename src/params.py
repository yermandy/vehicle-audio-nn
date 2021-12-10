from .constants import *
from easydict import EasyDict

def get_additional_params(config):
    n_samples_in_nn_hop = int(config.sr * config.nn_hop_length)
    n_samples_in_window = int(config.sr * config.window_length)
    n_features_in_sec = config.sr // config.hop_length
    n_features_in_nn_hop = int(n_features_in_sec * config.nn_hop_length)
    n_features_in_window = int(n_features_in_sec * config.window_length)
    config.n_samples_in_nn_hop = n_samples_in_nn_hop
    config.n_samples_in_window = n_samples_in_window
    config.n_features_in_nn_hop = n_features_in_nn_hop
    config.n_features_in_window = n_features_in_window
    return config