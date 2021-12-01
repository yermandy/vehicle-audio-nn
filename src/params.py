from .constants import *
from easydict import EasyDict


def get_params(**kwargs):
    # define parameters
    params = EasyDict()
    # hop length between nn inputs in features
    params.nn_hop_length = 6.0
    # length of one frame in seconds
    params.frame_length = 6.0
    # number of frames in one window
    params.n_frames = 1
    # length of one feature in samples
    params.n_fft = 1024
    # number of mel features
    params.n_mels = 64
    # number of mfcc features
    params.n_mfcc = 8
    # sampling rate
    params.sr = 44100
    # hop length between samples for feature extractor
    params.hop_length = 128
    # length of one window in seconds
    params.window_length = params.nn_hop_length * (params.n_frames - 1) + params.frame_length
    # normalization type
    params.normalization = Normalization.NONE
    # calculate additional params
    params = get_additional_params(params)
    # override params
    params.update(kwargs)
    return params


def get_additional_params(config):
    n_samples_in_nn_hop = int(config.sr * config.nn_hop_length)
    n_samples_in_frame = int(config.sr * config.frame_length)
    n_features_in_sec = config.sr // config.hop_length
    n_features_in_nn_hop = int(n_features_in_sec * config.nn_hop_length)
    n_features_in_frame = int(n_features_in_sec * config.frame_length)
    config.n_samples_in_nn_hop = n_samples_in_nn_hop
    config.n_samples_in_frame = n_samples_in_frame
    config.n_features_in_nn_hop = n_features_in_nn_hop
    config.n_features_in_frame = n_features_in_frame
    return config