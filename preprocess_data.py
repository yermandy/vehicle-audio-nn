import os
import sys
import numpy as np
import torch
import hydra
from src import *


os.makedirs('data/audio', exist_ok=True)
os.makedirs('data/labels', exist_ok=True)
os.makedirs('data/intervals', exist_ok=True)


def extract_audio(file):
    import moviepy.editor as mp
    video = mp.VideoFileClip(f"data/video/{file}.MP4")
    video.audio.write_audiofile(f"data/audio/{file}.MP4.wav")


def optimize(views, events_start_time, events_end_time, e_p_s, energy, is_rear=True, window_len=0.5):
    if is_rear:
        mask = views == 'rear'
        subset = events_start_time[mask] * e_p_s
    else:
        mask = views != 'rear'
        subset = events_end_time[mask] * e_p_s

    window_len = window_len * e_p_s

    delta_best = None
    sum_of_energies_best = 0

    deltas = np.arange(0, 5.1, 0.1)

    for delta in deltas:
        delta = delta * e_p_s

        if is_rear:
            window_from = subset - delta - window_len
        else:
            window_from = subset + delta - window_len
        window_from = np.clip(window_from, 0, len(energy)).astype(int)

        if is_rear:
            window_till = subset - delta + window_len
        else:
            window_till = subset + delta + window_len
        window_till = np.clip(window_till, 0, len(energy)).astype(int)

        sum_of_energies = 0

        for i, j in zip(window_from, window_till):
            sum_of_energies += energy[i: j].sum().item()

        if sum_of_energies > sum_of_energies_best:
            sum_of_energies_best = sum_of_energies
            delta_best = delta

    return delta_best / e_p_s


def extract_labels(file):
    signal, sr = load_audio(f'data/audio/{file}.MP4.wav', return_sr=True)
    signal_length = len(signal) // sr
    csv = load_csv(f'{file}.MP4')
    views = load_views_from_csv(csv)
    events_start_time, events_end_time = load_event_time_from_csv(csv)

    n_fft = 1024
    hop_length = 128

    s = torch.stft(signal, n_fft=n_fft, hop_length=hop_length)
    energy = s[..., 0].pow(2)
    energy = energy.sum(0)

    e_p_s = len(energy) / signal_length
    
    mask = views == 'rear'
    output_rear = optimize(views, events_start_time, events_end_time, e_p_s, energy, is_rear=True)
    estimated_labels_1 = events_start_time[mask] - output_rear

    mask = views != 'rear'
    output_front = optimize(views, events_start_time, events_end_time, e_p_s, energy, is_rear=False)
    estimated_labels_2 = events_end_time[mask] + output_front
    
    print(f'{file}: {output_rear:.2f}, {output_front:.2f}')
    
    estimated_labels = np.concatenate([estimated_labels_1, estimated_labels_2])
    estimated_labels.sort()
    estimated_labels = np.clip(estimated_labels, 0, signal_length)
    estimated_labels = np.round(estimated_labels, 2)
    estimated_labels.tolist()
    
    np.savetxt(f'data/labels/{file}.MP4.txt', estimated_labels, fmt='%s')


def extract_intevals(file, empty_interval_in_s=10):
    labels_file = f'data/labels/{file}.MP4.txt'
    csv = load_csv(f'{file}.MP4')
    events_start_times = csv[:, 8]
    events_end_times = csv[:, 9]

    events_start_times = np.array([time_to_sec(e) for e in events_start_times if e != ''])
    events_end_times = np.array([time_to_sec(e) for e in events_end_times if e != ''])
    # fill space between start and end time
    events = np.linspace(events_start_times, events_end_times, num=100).flatten()
    events = np.sort(events)
    
    next_cut = 0
    intervals = []
    for i in range(1, len(events)):
        diff = events[i] - events[i - 1]
        if diff > empty_interval_in_s:
            cut_at = events[i] - diff / 2
            intervals.append([f'{next_cut:.2f}', f'{cut_at:.2f}'])
            next_cut = cut_at
    
    intervals_file = f'data/intervals/{file}.MP4.txt'
    np.savetxt(intervals_file, intervals, fmt='%s')
    print(labels_file, len(intervals))


@hydra.main(config_path='config', config_name='config')
def preprocess(config):
    files = config.dataset
    for file in files:
        print('File: ', file)
        print('\nExtracting audio')
        extract_audio(file)
        print('\nExtracting labels')
        extract_labels(file)
        print('\nExtracting intervals')
        extract_intevals(file)


if __name__ == '__main__':
    
    sys.argv.append(f'hydra.run.dir=.')
    sys.argv.append(f'hydra.output_subdir=null')
    sys.argv.append(f'hydra/job_logging=disabled')
    sys.argv.append(f'hydra/hydra_logging=none')

    preprocess()