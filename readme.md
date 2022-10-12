## Audio-Based Event Detection

#### Experiments

[📖 Notion page](https://www.notion.so/yermandy/Audio-Based-Event-Detection-840a4b52f9a04aaf9f017610c4a7c91e)

#### Packages

Install with pip
``` 
hydra-core==1.1.1
wandb==0.12.10
librosa==0.8.1
tqdm
easydict
moviepy
tabulate
```

:warning: Install torch with pip, not conda

``` bash
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```


#### Project structure

Create and populate `data/video` and `data/csv` folders
``` bash
mkdir -p data/video
mkdir -p data/csv

# cvut dataset
mkdir -p data/video/cvut
mkdir -p data/csv/cvut
ln -s ~/data/MultiDo/CVUTFD/copy/*.{MP4,MOV,mov,mp4} data/video/
ln -s ~/data/MultiDo/CVUTFD/result/*.csv data/csv/

# eyedea dataset
mkdir -p data/video/eyedea
mkdir -p data/csv/eyedea
ln -s ~/data/MultiDo/videa_prujezdy/*.{MP4,MOV,mov,mp4} data/video/
ln -s ~/data/MultiDo/videa_prujezdy/*.csv data/csv/
```

#### Preprocess files

Use `preprocess_data.py` to generate files in `data/audio`, `data/audio_tensors`, `data/labels` and `data/intervals` 

Example:

``` bash
preprocess_data.py config/dataset/dataset.yaml
```

where `config/dataset/dataset.yaml` is the path to yaml list with files to be preprocessed

Converting videos by ffmpeg:
``` bash
ffmpeg -i input_video.mts -c:v copy -c:a aac -b:a 256k output_video.mp4
```


#### Wandb account

To visualize training curves, create [wandb](https://wandb.ai/) account and add new project. Add your wandb project name and account name to `config/wandb/wandb.yaml`.

#### Neural Network Training

Change training configurations in `config/model/default.yaml`

To override run configuration, use:
``` bash
python cross_validation.py experiment=047_october
```

where `047_october` is the name of the experiment defined in `config/experiment/047_october.yaml` file

