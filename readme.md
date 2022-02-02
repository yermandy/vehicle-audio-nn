## Audio-Based Event Detection

#### Packages

Install with pip
```
hydra-core==1.1.1
wandb==0.12.7
librosa==0.8.1
tqdm
easydict
moviepy
tabulate
```

Install torch with pip
```
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```


#### Project structure

Create and populate `data/video` and `data/csv` folders
```
mkdir -p data/video
mkdir -p data/csv
ln -s ~/data/MultiDo/CVUTFD/copy/* data/video/
ln -s ~/data/MultiDo/CVUTFD/result/* data/csv/
```

#### Preprocess files

Use `preprocess_data.py` to generate `data/audio`, `data/audio` and `data/intervals` 

Notice, `config/config.yaml` defines the dataset which will be processed

```
defaults:
 - dataset: dataset_26.11.2021 
```

where `dataset_26.11.2021` is the list of files which are going to be preprocessed and is placed in

```
./config/dataset/dataset_26.11.2021.yaml
````

```

Converting videos by ffmpeg:
$ ffmpeg -i input_video.mts -c:v copy -c:a aac -b:a 256k output_video.mp4
````


#### Wandb account

To visualize training curves, create [wandb](https://wandb.ai/) account and add new project. Add your wandb project name and account name to `config/config.yaml`.

#### Training

Change training configurations in `config/config.yaml` and run `train_classification.py`

Specify another config file
```
python train_classification.py --config-name config_name
python cross_validation.py --config-name config_name
```

#### Resources

* [YouTube: Audio Signal Processing for Machine Learning](https://www.youtube.com/playlist?list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0)

* [GitHub: Audio Signal Processing for Machine Learning](https://github.com/musikalkemist/AudioSignalProcessingForML)

* [YouTube: PyTorch for Audio and Music Processing](https://www.youtube.com/playlist?list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm)

* [Code: Robust audio-based vehicle counting in low-to-moderate traffic flow](http://cmp.felk.cvut.cz/data/audio_vc/)

#### Papers

* [Robust Audio-Based Vehicle Counting in Low-to-Moderate Traffic Flow](https://arxiv.org/pdf/2010.11716.pdf)