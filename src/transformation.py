import torch
import torch.nn as nn
import torchaudio
import torchvision
import torchaudio.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms as TV

from .constants import *


def create_stftkwargs(config):
    return {
        'n_fft': config.n_fft, 
        'hop_length': config.hop_length,
        'power': 1
    }

def create_melkwargs(config):
    return {
        'n_mels': config.n_mels,
        'f_min': config.f_min,
        'f_max': config.f_max
    }


def create_mel_transform(config):
    return T.MelSpectrogram(sample_rate=config.sr, **create_melkwargs(config), **create_stftkwargs(config))


def create_mfcc_transform(config):
    return T.MFCC(sample_rate=config.sr, n_mfcc=config.n_mfcc, melkwargs=create_melkwargs(config))


def create_stft_transform(config):
    return T.Spectrogram(**create_stftkwargs(config))


def create_feature_augmentations(config):
    return nn.Sequential(
        T.TimeMasking(time_mask_param=config.time_mask_param),
        T.FrequencyMasking(freq_mask_param=config.freq_mask_param)
    )


def initialize(config):
    if 'f_max' not in config:
        config.f_max = None

    if 'f_min' not in config:
        config.f_min = 0

    if 'n_mfcc' not in config:
        config.n_mfcc = 0

    if 'feature_augmentation' not in config:
        config.feature_augmentation = False

    if 'gaussian_blur' not in config:
        config.gaussian_blur = False

    if 'image_augmentations' not in config:
        config.image_augmentations = False

    if 'resize' not in config:
        config.resize = False

    if 'resize_size' not in config:
        config.resize_size = [64, 64]

    if 'transformation' not in config:
        config.transformation = 'mel'


def create_transformation(config, part: Part = Part.TEST):
    initialize(config)

    use_mfcc = config.transformation == 'mfcc'
    use_mel = config.transformation == 'mel'
    use_stft = config.transformation == 'stft'
    use_augmentations = config.feature_augmentation
    use_gaussian_blur = config.gaussian_blur
    use_image_augmentations = config.image_augmentations

    normalization = Normalization(config.normalization)

    # apply logarithmic compression: https://arxiv.org/pdf/1709.01922.pdf
    amplitude_to_DB = T.AmplitudeToDB('energy')
    
    if use_mel:
        mel_transform = create_mel_transform(config)

    if use_stft:
        stft_transform = create_stft_transform(config)
        
    if use_mfcc:
        mfcc_transform = create_mfcc_transform(config)

    if use_augmentations:
        augmentations = create_feature_augmentations(config)

    if use_gaussian_blur:
        gaussian_blur = lambda x: F.gaussian_blur(x, config.gaussian_blur_kernel_size, config.gaussian_blur_sigma)

    if config.resize:
        resize = TV.Resize(config.resize_size)

    if use_image_augmentations:
        image_augmentations = TV.RandomChoice([
            lambda x: x,
            TV.GaussianBlur(3),
            TV.GaussianBlur(5),
            TV.GaussianBlur(7)
        ])

    def transform(signal) -> torch.Tensor:
        if use_mfcc:
            features = mfcc_transform(signal)
        elif use_stft:
            features = stft_transform(signal)
            features = amplitude_to_DB(features)
        else:
            features = mel_transform(signal)
            features = amplitude_to_DB(features)

        if config.resize:
            features = resize(features.unsqueeze(0)).squeeze()
            
        if normalization.is_none():
            features = features.unsqueeze(0)
        elif normalization.is_global():
            # normalize globally
            normalize = lambda x: (x - x.mean()) / torch.maximum(x.std(), torch.tensor(1e-8))
            features = normalize(features)
            features = features.unsqueeze(0)
        elif normalization.is_row_wise():
            # normalize features row wise
            features = features.unsqueeze(0)
            features = (features - features.mean(2).view(-1, 1)) / torch.maximum(features.std(2).view(-1, 1), torch.tensor(1e-8))
        elif normalization.is_column_wise():
            # normalize features column wise
            normalize = lambda x: (x - x.mean(0)) / torch.maximum(x.std(0), torch.tensor(1e-8))
            features = normalize(features)
            features = features.unsqueeze(0)
        else:
            raise Exception('unknown normalization')

        if use_augmentations and part.is_trn():
            features = augmentations(features)

        if use_gaussian_blur:
            features = gaussian_blur(features)

        if use_image_augmentations and part.is_trn():
            features = image_augmentations(features)
            
        return features

    return transform
