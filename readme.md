## Audio-Based Event Detection

#### Packages

Unstall with pip
```
hydra-core==1.1.1
wandb==0.12.7
librosa==0.8.1
torch==1.9.0+cu111
torchaudio==0.9.0
torchvision==0.10.0+cu111
```

#### Project structure

Create and populate `video` and `csv` folders
```
mkdir -p data/video
mkdir -p data/csv
ln -s ~/data/MultiDo/CVUTFD/copy/* data/video/
ln -s ~/data/MultiDo/CVUTFD/result/* data/csv/
```

#### Resources

* [YouTube: Audio Signal Processing for Machine Learning](https://www.youtube.com/playlist?list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0)

* [GitHub: Audio Signal Processing for Machine Learning](https://github.com/musikalkemist/AudioSignalProcessingForML)

* [YouTube: PyTorch for Audio and Music Processing](https://www.youtube.com/playlist?list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm)

* [Code: Robust audio-based vehicle counting in low-to-moderate traffic flow](http://cmp.felk.cvut.cz/data/audio_vc/)

#### Papers

* [Robust Audio-Based Vehicle Counting in Low-to-Moderate Traffic Flow](https://arxiv.org/pdf/2010.11716.pdf)