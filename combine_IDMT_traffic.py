import pyrootutils.root
from src import *

files = glob("data/audio_wav/IDMT_Traffic/*.wav")

n_files = 0

np.random.shuffle(files)

audios = []

for file in files:
    if "-BG" in file:
        continue
    if "CH12" in file:
        continue
    print(file)
    audio, sr = torchaudio.load(file)
    audio = audio.mean(0)
    audio = T.Resample(sr, 44100).forward(audio)
    audios.append(audio)
    n_files += 1
    if n_files == 1000:
        break

audios = torch.hstack(audios)
sr = 44100

torch.save([audios, sr], "data/audio_pt/IDMT_Traffic/combined.pt")
print(audios.shape)
