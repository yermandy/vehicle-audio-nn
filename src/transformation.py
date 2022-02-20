import torch
import torch.nn as nn
import torchaudio
import torchvision
import torchaudio.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms as TV

from .constants import *


def create_melkwargs(config):
    return {"n_fft": config.n_fft, "n_mels": config.n_mels, "hop_length": config.hop_length}


def create_mel_transform(config):
    return T.MelSpectrogram(sample_rate=config.sr, 
            f_max=config.f_max,
            **create_melkwargs(config))


def create_mfcc_transform(config):
    return T.MFCC(sample_rate=config.sr, n_mfcc=config.n_mfcc, melkwargs=create_melkwargs(config))


def create_feature_augmentations(config):
    return nn.Sequential(
        T.TimeMasking(time_mask_param=config.time_mask_param),
        T.FrequencyMasking(freq_mask_param=config.freq_mask_param)
    )


def initialize(config):
    if 'f_max' not in config:
        config.f_max = None

    if 'n_mfcc' not in config:
        config.n_mfcc = 0

    if 'feature_augmentation' not in config:
        config.feature_augmentation = False

    if 'gaussian_blur' not in config:
        config.gaussian_blur = 0

    if 'image_augmentations' not in config:
        config.image_augmentations = False

    if 'resize' not in config:
        config.resize = False

    if 'resize_size' not in config:
        config.resize_size = [64, 64]


def create_transformation(config, is_train=False):
    initialize(config)

    use_mfcc = config.n_mfcc > 0
    use_augmentations = config.feature_augmentation
    use_resampling = config.sr != 44100
    use_gaussian_blur = config.gaussian_blur > 0
    use_image_augmentations = config.image_augmentations

    mel_transform = create_mel_transform(config)
    normalization = config.normalization
    amplitude_to_DB = T.AmplitudeToDB()
    
    if use_mfcc:
        mfcc_transform = create_mfcc_transform(config)

    if use_augmentations:
        augmentations = create_feature_augmentations(config)

    if use_resampling:
        resample = T.Resample(44100, config.sr)

    if use_gaussian_blur:
        gaussian_blur = lambda x: F.gaussian_blur(x, config.gaussian_blur, 0.1)
        # gaussian_blur = torchvision.transforms.GaussianBlur(config.gaussian_blur, sigma=0.1)

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
        if use_resampling:
            signal = resample(signal)
        
        if use_mfcc:
            features = mfcc_transform(signal)
        else:
            features = mel_transform(signal)
            # https://arxiv.org/pdf/1709.01922.pdf
            # apply logarithmic compression
            # features = torch.log10(features + 1)
            features = amplitude_to_DB(features)

        if config.resize:
            features = resize(features.unsqueeze(0)).squeeze()
            
        if normalization == Normalization.NONE:
            features = features.unsqueeze(0)
        elif normalization == Normalization.GLOBAL:
            # normalize globally
            normalize = lambda x: (x - x.mean()) / torch.maximum(x.std(), torch.tensor(1e-8))
            features = normalize(features)
            features = features.unsqueeze(0)
        elif normalization == Normalization.ROW_WISE:
            # normalize features row wise
            features = features.unsqueeze(0)
            features = (features - features.mean(2).view(-1, 1)) / torch.maximum(features.std(2).view(-1, 1), torch.tensor(1e-8))
        elif normalization == Normalization.COLUMN_WISE:
            # normalize features column wise
            normalize = lambda x: (x - x.mean(0)) / torch.maximum(x.std(0), torch.tensor(1e-8))
            features = normalize(features)
            features = features.unsqueeze(0)
        else:
            raise Exception('unknown normalization')

        if use_augmentations and is_train:
            features = augmentations(features)

        if use_gaussian_blur:
            features = gaussian_blur(features)

        if use_image_augmentations and is_train:
            features = image_augmentations(features)
            
        return features

    return transform
