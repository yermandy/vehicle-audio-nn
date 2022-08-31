import pyrootutils.root

from typing import Dict
from src.constants import *
from easydict import EasyDict
from dataclasses import dataclass, field
import omegaconf


@dataclass
class StructuredPredictor:
    # root with training files
    root: str = "outputs/000_structured_rvce/036"
    # split number
    split: int = 0
    # bmrm regularization constant
    reg: float = 10
    # bmrm relative tolerance
    tol_rel: float = 0.01
    # combine training and validation files
    combine_trn_and_val: bool = False
    # normalize features
    normalize_X: bool = False
    # seed for reproducibility
    seed: int = 42
    # number of events in small window
    Y: int = 7
    # learn only biases
    biases_only: bool = False
    # path for outputs
    outputs_folder: str = "outputs/036_results"

    training_files: str = None
    validation_files: str = None
    testing_files: str = None


@dataclass
class FeaturesConfig:
    # sampling rate [samples]
    sr: int = 22050

    # signal to feature transformation
    # stft | mel | mfcc
    transformation: Transformation = Transformation.STFT

    log_transformation: bool = True

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

    # image augmentations for training
    image_augmentations: bool = False
    random_gaussian_blur: bool = False
    random_erasing: bool = False
    random_resized_crop: bool = False

    # audio augmentations for training
    audio_augmentations: bool = False
    random_colored_noise: bool = False
    random_pitch_shift: bool = False
    random_gain: bool = False
    random_low_pass_filter: bool = False
    random_high_pass_filter: bool = False

    # zero mean, unit variance feature normalization:
    # none | row-wise | column-wise | global
    normalization: Normalization = Normalization.GLOBAL

    # zero mean, unit variance signal normalization
    signal_normalization: bool = False


@dataclass
class ModelConfig:
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
    # use random offset each time
    use_random_offset: bool = False
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
    # heads: Dict[str, float] = {'n_counts' : 1.0}
    heads: Dict[str, float] = None

    # inference type
    inference_function = InferenceFunction.SIMPLE

    # if inference is dense – specify the number overpalling windows
    n_windows_for_dense_inference: int = 3
    n_events_per_dense_window: int = 5

    # if inference is structured - specify which labels are coupled
    coupled_labels: list = None

    raw_signal: bool = False

    # architecture:
    # ResNet18 | WaveCNN
    architecture: str = "ResNet18"

    # optimizer
    # Adam | AdamW
    optimizer: str = "AdamW"

    rawnet_layers: list = None
    rawnet_filters: list = None

    loss: str = "CrossEntropy"

    loss_cbce_beta: float = 0.999

    # Transormer parameters
    transformer_dim: int = 1024
    transformer_patch_size: int = 16
    transformer_depth: int = 3
    transformer_heads: int = 10
    transformer_mlp_dim: int = 512
    transformer_dropout: float = 0.1
    transformer_emb_dropout: float = 0.1


@dataclass
class WandbConfig:
    wandb_project: str = None
    wandb_entity: str = None
    wandb_tags: tuple = None


@dataclass
class CrossValidation:
    splits_from: int = 0
    n_splits: int = 0
    cross_validation_folder: str = None


@dataclass
class Config(
    EasyDict, FeaturesConfig, ModelConfig, WandbConfig, CrossValidation, object
):
    uuid: str = None
    seed: int = 42
    n_samples_in_nn_hop: int = None
    n_samples_in_window: int = None
    training_files: tuple = ("12_RX100",)
    testing_files: tuple = ("12_RX100",)
    validation_files: tuple = ("12_RX100",)
    structured_predictor: StructuredPredictor = None

    use_testing_files: bool = True
    use_manual_counts: bool = False

    def __init__(self, config=None, **kwargs):
        super(Config, self).__init__(config, **kwargs)
        self.n_samples_in_nn_hop = int(self.sr * self.nn_hop_length)
        self.n_samples_in_window = int(self.sr * self.window_length)
        self.normalization = Normalization(self.normalization)
        self.transformation = Transformation(self.transformation)
        self.inference_function = InferenceFunction(self.inference_function)
        self.to_primitive()

    def to_primitive(self):
        # TODO recursive call on lists and dicts
        for k, v in self.items():
            if isinstance(v, omegaconf.ListConfig):
                self[k] = list(v)

    def set_nn_hop_length(self, nn_hop_length):
        self.nn_hop_length = nn_hop_length
        self.n_samples_in_nn_hop = int(self.sr * self.nn_hop_length)

    def set_window_length(self, window_length):
        self.window_length = window_length
        self.n_samples_in_window = int(self.sr * self.window_length)

    def __str__(self) -> str:
        return super().__str__()


if __name__ == "__main__":
    config = Config()
    print(config)
