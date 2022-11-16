## Audio-Based Event Detection

### Experiments

[📖 Notion page](https://www.notion.so/yermandy/Audio-Based-Event-Detection-840a4b52f9a04aaf9f017610c4a7c91e)

### Packages

#### Easy way

``` bash
conda env create --file environment.yml
```

#### Manual way

:warning: **Note**: This is not the recommended way. Use the `environment.yml` file instead to create the environment.

``` bash
conda create -n eye-audio python=3.10 
conda activate eye-audio
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install easydict 
pip install opencv-python
pip install tqdm
pip install matplotlib
pip install scikit-learn
pip install PyYAML
pip install wandb
pip install hydra-core
pip install pandas
pip install seaborn
pip install librosa
pip install moviepy
pip install tabulate
pip install git+https://github.com/yermandy/pyrootutils.git
pip install rich
pip install torch_audiomentations
pip install audiomentations
pip install qpsolvers[open_source_solvers]
```

### Project structure

Create and populate `data/video` and `data/csv` folders
``` bash
mkdir -p data/video
mkdir -p data/csv

# cvut dataset
mkdir -p data/video/cvut
mkdir -p data/csv/cvut
ln -s ~/data/MultiDo/CVUTFD/copy/*.{MP4,MOV,mov,mp4,mts,MTS} data/video/cvut/
ln -s ~/data/MultiDo/CVUTFD/result/*.csv data/csv/cvut/

# eyedea dataset
mkdir -p data/video/eyedea
mkdir -p data/csv/eyedea
ln -s ~/data/MultiDo/videa_prujezdy/*.{MP4,MOV,mov,mp4,mts,MTS} data/video/eyedea/
ln -s ~/data/MultiDo/videa_prujezdy/*.csv data/csv/eyedea/
```

### Preprocess files

Use `preprocess_data.py` to generate files in `data/audio`, `data/audio_tensors`, `data/labels` and `data/intervals` 

Example:

``` bash
preprocess_data.py config/dataset/000_debug.yaml
```

where `config/dataset/dataset.yaml` is the path to yaml list with files to be preprocessed

Converting videos by ffmpeg:
``` bash
ffmpeg -i input_video.mts -c:v copy -c:a aac -b:a 256k output_video.mp4
```


### Wandb account

To visualize training curves, create [wandb](https://wandb.ai/) account and add new project. Add your wandb project name and account name to `config/wandb/wandb.yaml`.

### Neural Network Training

#### Debug training

The following command will run training for a few epochs and save results to `outputs/000_debug` folder

``` bash
python cross_validation.py experiment=000_debug cuda=1
```

#### Best Model

Change training configurations in `config/model/default.yaml`

To override run configuration, use:
``` bash
python cross_validation.py experiment=047_october
```

where `047_october` is the name of the experiment defined in `config/experiment/047_october.yaml` file


### Weights

Download pretrained model [here](https://drive.google.com/file/d/1v6vbDJDzXYF-nHO7PFSXa3hgL8ttfppY) and unzip in `outputs` folder

### Demos

#### Demo 1

Prediction.

It takes an audio, extracted from a video, applies multi-head audio predictor and outputs predictions for individual time windows and summary. 

Input:
1. videos
2. model

Output:
1. predictions for each time window
2. counts for each head

Usage:

``` bash
python demo_1.py -v 71_Samsung -m 047_october/0
```

Notice, `71_Samsung` video file should be somewhere in subdirectories of `data/video/**`. The full model path is "outputs/047_october/0/rvce.pth". 

#### Demo 2

Prediction and evaluation. The same as demo_1, but it uses ground-truth labels to evaluate prediction accuracy.

Input:
1. videos
2. model
3. csv files with annotations

Output:
1. rvce for each head
2. fault detection visualization

Usage:

``` bash
python demo_2.py -v 71_Samsung -m 047_october/0
```

Notice, `71_Samsung` video file should be somewhere in subdirectories of `data/video/**` and annotations in `data/csv/**`. The full model path is "outputs/047_october/0/rvce.pth". 

#### Demo 3

It splits input (long) video into two parts. The begining part is used for fine-tuning the prediction model. The trailing part of the video is used for prediction and evaluation.

Input:
1. videos
2. model
3. csv files with annotations
4. fine-tuning length (training part)

Output:
1. rvce for each head on test part

Usage:

``` bash
python demo_3.py -v 71_Samsung -m 047_october/0
```

Notice, `71_Samsung` video file should be somewhere in subdirectories of `data/video/**` and annotations in `data/csv/**`. The full model path is "outputs/047_october/0/rvce.pth". 
