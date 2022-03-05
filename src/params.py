from dataclasses import dataclass
from .constants import *
from easydict import EasyDict
from dataclasses import dataclass



@dataclass
class FeaturesConfig():
    # sampling rate [samples]
    sr: int = 22050
    
    # stft | mel | mfcc
    transformation: str = 'stft'

    #! stft parameters
    # size of FFT, creates n_fft // 2 + 1 bins
    n_fft: int = 1024
    # length of hop between STFT windows
    hop_length: int = 512

    #! mel parameters (uses stft parameters)
    # number of mel features
    n_mels: int = 128
    # frequency upper bound: int or null
    f_max: int = 11000
    # frequency lower bound
    f_min: int = 0

    #! mfcc parameters (uses stft and mel parameters)
    # number of mfcc features
    n_mfcc: int = 0

    #! features post processing
    # time masking and frequency masking augmentations for training
    feature_augmentation: bool = False
    time_mask_param: int = None
    freq_mask_param: int = None

    # size of gaussian kernel and sigma e.g.: [5, 0.5]
    gaussian_blur: bool = False
    gaussian_blur_kernel_size: int = None
    gaussian_blur_sigma: float = None

    # resize features
    resize: bool = False
    # resize to size e.g.: [128, 128]
    resize_size: tuple = (128, 128)

    # zero mean, unit variance feature normalization:
    # none | row-wise | column-wise | global
    normalization: str = 'global'


@dataclass
class ModelConfig(object):
    # gpu number
    cuda: int = 0
    # learning rate
    lr: float = 0.0001
    # number of traing epochs
    n_epochs: int = 200
    # batch size
    batch_size: int = 64
    # number of workers
    num_workers: int = 8
    # training to validation ratio
    split_ratio: float = 0.75
    # use new offset after each epoch
    use_offset: bool = True
    # offset length in sec
    offset_length: float = 0.25
    # number of classes to predict
    num_classes: int = 50

    # length of one hop in seconds
    nn_hop_length: float = 6.0
    # length of one window in seconds
    window_length: float = 6.0
    # number of frames in one window

    # neural network heads
    heads: tuple = ('n_counts')


@dataclass
class WandbConfig:
    wandb_project: str
    wandb_entity: str
    wandb_tags: tuple


@dataclass
class Config(EasyDict, FeaturesConfig, ModelConfig, WandbConfig, object):
    uuid: 'str' = None
    n_samples_in_nn_hop: int = None
    n_samples_in_window: int = None

    def __init__(self, config=None, **kwargs):
        self.n_samples_in_nn_hop = int(self.sr * self.nn_hop_length)
        self.n_samples_in_window = int(self.sr * self.window_length)
        super(Config, self).__init__(config, **kwargs)

    def __str__(self) -> str:
        return super().__str__()