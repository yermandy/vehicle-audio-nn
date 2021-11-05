import torch
import torchaudio
import librosa
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as tick

import warnings
warnings.filterwarnings("ignore")


def get_melkwargs(params):
    return {
        "n_fft": params.n_fft,
        "n_mels": params.n_mels,
        "hop_length": params.hop_length
    }

def conv(array):
    array = array.view(1, 1, -1)
    kernel = torch.ones(1, 1, 201)
    array = torch.nn.functional.conv1d(array, kernel)
    array = array.squeeze()
    return array

def show(params, signal, events=None, 
         predictions=None,
         probabilities=None,
         events_start_time=None, events_end_time=None, 
         manual_events=None, directions=None, views=None,
         from_time=0, till_time=86400,
         save=None):
    
    print(f'{from_time // 60:02.0f}:{from_time % 60:02.0f} - {till_time // 60:02.0f}:{till_time % 60:02.0f}')
    
    signal = signal[from_time * params.sr: till_time * params.sr]
    
    signal_length = len(signal) / params.sr

    nrows = 3
    if predictions is not None:
        nrows += 1 
    width = (till_time - from_time) // 4
    height = 4 * nrows
    fig, axes = plt.subplots(nrows=nrows, figsize=(width, height))
    
    if predictions is not None:
        ax0, ax3, ax1, ax2 = axes        
    else:
        ax0, ax1, ax2 = axes
    
    for ax in axes:
        ax.margins(0, 0.02)
        
    x_axis = np.arange(from_time, till_time + 1)
    ax0.plot(x_axis, np.zeros(len(x_axis)), marker='o', markersize=3, color='black')
    
    def formatter(x, y):
        return f'{x // 60:02.0f}:{x % 60:02.0f}'

    ax0.xaxis.set_major_formatter(tick.FuncFormatter(formatter))
    ax0.set_xticks(np.arange(x_axis[0], x_axis[-1] + 1, 10.0))
    
    # show events from eyedea engine
    if events is not None:
        colors = 'violet'
        mask = (events >= from_time) & (events < till_time)
        
        # color code direction
        if directions is not None:
            colors = ['red' if direction == 'outgoing' else 'green' for direction in directions[mask]]
        
        # color code direction
        if views is not None:
            colors = ['red' if view == 'rear' else 'green' for view in views[mask]]

        ax0.vlines(events[mask], 0, 1, color=colors, linewidth=2.0)
                 
    # show manual annotations
    if manual_events is not None:
        mask = (manual_events >= from_time) & (manual_events < till_time)
        ax0.vlines(manual_events[mask], 0, 1, color='black', linestyle=':', linewidth=2.0)
        
    # show start and end time of events
    if events_start_time is not None and events_end_time is not None:
        colors = 'violet'
        
        # color code direction
        if directions is not None:
            colors = ['red' if direction == 'outgoing' else 'green' for direction in directions]
        
        # color code direction
        if views is not None:
            colors = ['red' if view == 'rear' else 'green' for view in views]
        
        for event_start_time, event_end_time, color in zip(events_start_time, events_end_time, colors):
            if from_time <= event_start_time <= till_time and from_time <= event_end_time <= till_time:
                ax0.fill_between([event_start_time, event_end_time], [0], [1], color=color, alpha=0.25);                

    # plot predictions
    if predictions is not None:
        if signal_length % params.window_length != 0:
            print(f'interval is not divisible by {params.window_length}')
        ax3.xaxis.set_major_formatter(tick.FuncFormatter(formatter))
        ax3.set_xticks(np.arange(x_axis[0], x_axis[-1] + 1, 10.0))
        from .utils import get_time
        x_axis_time = get_time(signal, params, from_time, till_time)
        #! introduce dummy ending
        predictions = np.append(predictions, 0)
        ax3.step(x_axis_time, predictions, where='post', linewidth=3.0, c='tab:red')
        
        max_output = int(np.max(predictions))
        ax3.hlines(np.arange(1, max_output + 1), x_axis_time[0], x_axis_time[-1], color='k', linestyle='dotted', linewidth=1.0)
        max_output = max(max_output, 1)
        ax3.vlines(x_axis_time, 0, max_output, color='k', linestyle='dotted', linewidth=1, alpha=0.5)
        
        if probabilities is not None:
            x_axis_time = get_time(signal, params, from_time, till_time)
            for x, p in zip(x_axis_time[:-1], probabilities):
                predicted_class = np.argmax(p)
                for i, p_i in zip(range(predicted_class + 1), p):
                    ax3.text(x + 0.5, i + 0.25, f'{i} : {p_i:.4f}', fontsize=13)    
                # ax3.text(x + 0.5, 0.25, f'{np.argmax(p)} : {np.max(p):.4f}', fontsize=13)

        if events is not None:
            events_in_windows = []
            x_axis_time = get_time(signal, params, from_time, till_time)
            for i in range(1, len(x_axis_time)):
                events_in_window = (manual_events >= x_axis_time[i - 1]) & (manual_events < x_axis_time[i])
                events_in_windows.append(events_in_window.sum())
            #! introduce dummy ending
            events_in_windows.append(0)
            ax3.step(x_axis_time, events_in_windows, where='post', linewidth=3.0, linestyle='dotted', c='black')

    # plot signal amplitude
    each = 16
    ax1.plot(signal[::each], alpha=0.5)
    
    features = torch.stft(signal, n_fft=params.n_fft, hop_length=params.hop_length)
    energy = features[..., 0].pow(2)
    energy = energy.sum(0)
    energy = conv(energy)
    energy -= energy.min()
    energy /= energy.max()

    # plot signal energy
    x_axis = np.linspace(0, len(signal[::each]), len(energy))
    ax1_1 = ax1.twinx()
    ax1_1.plot(x_axis, energy, c='black')

    # plot spectrogram
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=params.sr, **get_melkwargs(params))
    features = transform(signal).numpy()
    features = librosa.power_to_db(features, ref=np.max)
    ax2.pcolormesh(features)
    
    # plot approximate position of highest energy
#     if events_start_time is not None and events_end_time is not None and views is not None:
#         lines = []
#         for event_start_time, event_end_time, view in zip(events_start_time, events_end_time, views):

#             # rear
#             event_start_time = event_start_time - 2.9
#             # front
#             event_end_time = event_end_time + 1.4

#             if from_time <= event_start_time <= till_time and from_time <= event_end_time <= till_time:
#                 if view == 'rear':
#                     lines.append(event_start_time)
#                 else:
#                     lines.append(event_end_time)

#         ax0.vlines(lines, 0, 1, color='magenta', linestyle=':', linewidth=2.0)
    
    if save is not None and save is not False:
        plt.tight_layout()
        plt.savefig(save, dpi=100)
        
    plt.show()

# params = EasyDict()
# params.n_fft = 1024
# params.n_mels = 64
# params.hop_length = 128
# params.sr = 44100

# params.nn_hop_length = 1
# params.frame_length = 1