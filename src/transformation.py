import torch
import torch.nn as nn
import torchaudio
import torchvision
import torchaudio.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms as TV
import torch_audiomentations as TA
from .config import Config

from .constants import *
from audiomentations import AddGaussianNoise, PitchShift


def create_stftkwargs(config: Config):
    return {"n_fft": config.n_fft, "hop_length": config.hop_length, "power": 1}


def create_melkwargs(config: Config):
    return {"n_mels": config.n_mels, "f_min": config.f_min, "f_max": config.f_max}


def create_mel_transform(config: Config):
    return T.MelSpectrogram(
        sample_rate=config.sr, **create_melkwargs(config), **create_stftkwargs(config)
    )


def create_mfcc_transform(config: Config):
    return T.MFCC(
        sample_rate=config.sr, n_mfcc=config.n_mfcc, melkwargs=create_melkwargs(config)
    )


def create_stft_transform(config: Config):
    return T.Spectrogram(**create_stftkwargs(config))


def create_feature_augmentations(config: Config):
    return nn.Sequential(
        T.TimeMasking(time_mask_param=config.time_mask_param),
        T.FrequencyMasking(freq_mask_param=config.freq_mask_param),
    )


def create_image_augmentations(config: Config):
    transforms = []
    if config.random_gaussian_blur:
        transforms.append(TV.RandomApply([TV.GaussianBlur(3)], p=0.5))
    if config.random_erasing:
        # applies with probability 0.5 by default
        transforms.append(TV.RandomErasing(inplace=True))
    if config.random_resized_crop:
        transforms.append(
            TV.RandomApply([TV.RandomResizedCrop(config.resize_size)], p=0.5)
        )
    return TV.RandomChoice(transforms)


def create_audio_augmentations(config: Config):
    transforms = []
    if config.random_colored_noise:
        transforms.append(TA.AddColoredNoise())
    if config.random_pitch_shift:
        transforms.append(TA.PitchShift(config.sr))
    if config.random_gain:
        transforms.append(TA.Gain())
    if config.random_low_pass_filter:
        transforms.append(TA.LowPassFilter())
    if config.random_high_pass_filter:
        transforms.append(TA.HighPassFilter())
    # return TA.Compose(transforms)
    return TA.OneOf(transforms)


def create_transformation(config: Config, is_trn=False):
    # apply logarithmic compression: https://arxiv.org/pdf/1709.01922.pdf
    amplitude_to_DB = T.AmplitudeToDB("energy")

    if config.transformation.is_mel():
        mel_transform = create_mel_transform(config)

    if config.transformation.is_stft():
        stft_transform = create_stft_transform(config)

    if config.transformation.is_mfcc():
        mfcc_transform = create_mfcc_transform(config)

    if config.feature_augmentation:
        augmentations = create_feature_augmentations(config)

    if config.gaussian_blur:
        gaussian_blur = lambda x: F.gaussian_blur(
            x, config.gaussian_blur_kernel_size, config.gaussian_blur_sigma
        )

    if config.resize:
        resize = TV.Resize(config.resize_size)

    if config.image_augmentations:
        image_augmentations = create_image_augmentations(config)

    if config.audio_augmentations:
        audio_augmentations = create_audio_augmentations(config)

    def transform(signal) -> torch.Tensor:

        if config.audio_augmentations and is_trn:
            signal = signal.reshape(1, 1, -1)
            signal = audio_augmentations(signal, config.sr)
            signal = signal.squeeze()

        if config.raw_signal:
            return signal.unsqueeze(0)

        if config.transformation.is_mfcc():
            features = mfcc_transform(signal)
        elif config.transformation.is_stft():
            features = stft_transform(signal)
            if config.log_transformation:
                features = amplitude_to_DB(features)
        elif config.transformation.is_mel():
            features = mel_transform(signal)
            if config.log_transformation:
                features = amplitude_to_DB(features)
        else:
            raise Exception("unknown transformation")

        if config.resize:
            features = resize(features.unsqueeze(0)).squeeze()

        if config.normalization.is_none():
            features = features.unsqueeze(0)
        elif config.normalization.is_global():
            # normalize globally
            normalize = lambda x: (x - x.mean()) / torch.maximum(
                x.std(), torch.tensor(1e-8)
            )
            features = normalize(features)
            features = features.unsqueeze(0)
        elif config.normalization.is_row_wise():
            # normalize features row wise
            features = features.unsqueeze(0)
            features = (features - features.mean(2).view(-1, 1)) / torch.maximum(
                features.std(2).view(-1, 1), torch.tensor(1e-8)
            )
        elif config.normalization.is_column_wise():
            # normalize features column wise
            normalize = lambda x: (x - x.mean(0)) / torch.maximum(
                x.std(0), torch.tensor(1e-8)
            )
            features = normalize(features)
            features = features.unsqueeze(0)
        else:
            raise Exception("unknown normalization")

        if config.feature_augmentation and is_trn:
            features = augmentations(features)

        if config.gaussian_blur and is_trn:
            features = gaussian_blur(features)

        if config.image_augmentations and is_trn:
            features = image_augmentations(features)

        return features

    return transform
