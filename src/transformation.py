import torch
import torch.nn as nn
import torchaudio
import torchvision
import torchaudio.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms as TV
from .config import Config

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


def create_transformation(config: Config, part: Part = Part.TEST):
    # apply logarithmic compression: https://arxiv.org/pdf/1709.01922.pdf
    amplitude_to_DB = T.AmplitudeToDB('energy')
    
    if config.transformation.is_mel():
        mel_transform = create_mel_transform(config)

    if config.transformation.is_stft():
        stft_transform = create_stft_transform(config)
        
    if config.transformation.is_mfcc():
        mfcc_transform = create_mfcc_transform(config)

    if config.feature_augmentation:
        augmentations = create_feature_augmentations(config)

    if config.gaussian_blur:
        gaussian_blur = lambda x: F.gaussian_blur(x, config.gaussian_blur_kernel_size, config.gaussian_blur_sigma)

    if config.resize:
        resize = TV.Resize(config.resize_size)

    if config.image_augmentations:
        image_augmentations = TV.RandomChoice([
            lambda x: x,
            TV.GaussianBlur(3),
            TV.GaussianBlur(5),
            TV.GaussianBlur(7)
        ])

    def transform(signal) -> torch.Tensor:
        if config.transformation.is_mfcc():
            features = mfcc_transform(signal)
        elif config.transformation.is_stft():
            features = stft_transform(signal)
            features = amplitude_to_DB(features)
        elif config.transformation.is_mel():
            features = mel_transform(signal)
            features = amplitude_to_DB(features)
        else:
            raise Exception('unknown transformation')

        if config.resize:
            features = resize(features.unsqueeze(0)).squeeze()
            
        if config.normalization.is_none():
            features = features.unsqueeze(0)
        elif config.normalization.is_global():
            # normalize globally
            normalize = lambda x: (x - x.mean()) / torch.maximum(x.std(), torch.tensor(1e-8))
            features = normalize(features)
            features = features.unsqueeze(0)
        elif config.normalization.is_row_wise():
            # normalize features row wise
            features = features.unsqueeze(0)
            features = (features - features.mean(2).view(-1, 1)) / torch.maximum(features.std(2).view(-1, 1), torch.tensor(1e-8))
        elif config.normalization.is_column_wise():
            # normalize features column wise
            normalize = lambda x: (x - x.mean(0)) / torch.maximum(x.std(0), torch.tensor(1e-8))
            features = normalize(features)
            features = features.unsqueeze(0)
        else:
            raise Exception('unknown normalization')

        if config.feature_augmentation and part.is_trn():
            features = augmentations(features)

        if config.gaussian_blur:
            features = gaussian_blur(features)

        if config.image_augmentations and part.is_trn():
            features = image_augmentations(features)
            
        return features

    return transform
