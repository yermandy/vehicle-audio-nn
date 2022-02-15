import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

from .constants import *


def create_melkwargs(config):
    return {"n_fft": config.n_fft, "n_mels": config.n_mels, "hop_length": config.hop_length}


def create_mel_transform(config):
    return T.MelSpectrogram(sample_rate=config.sr, **create_melkwargs(config))


def create_mfcc_transform(config):
    return T.MFCC(sample_rate=config.sr, n_mfcc=config.n_mfcc, melkwargs=create_melkwargs(config))


def create_feature_augmentations(config):
    return nn.Sequential(
        T.TimeMasking(time_mask_param=config.time_mask_param),
        T.FrequencyMasking(freq_mask_param=config.freq_mask_param)
    )


def create_transformation(config, is_train=False):
    use_mfcc = True if 'n_mfcc' in config and config.n_mfcc is not None and config.n_mfcc > 0 else False
    use_augmentations = True if 'feature_augmentation' in config and config.feature_augmentation is True else False
    use_resampling = True if config.sr != 44100 else False

    mel_transform = create_mel_transform(config)
    
    if use_mfcc:
        mfcc_transform = create_mfcc_transform(config)

    if use_augmentations:
        augmentations = create_feature_augmentations(config)

    amplitude_to_DB = T.AmplitudeToDB()

    if use_resampling:
        resample = T.Resample(44100, config.sr)
    
    normalization = config.normalization

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

        return features

    return transform
