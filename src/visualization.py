import torch
import torchaudio
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as tick

import warnings

from .transformation import create_transformation
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

def show_video(file, scale=0.3):
    from IPython.display import HTML
    return HTML(f"""
        <video width="{1920 * scale}" height="{1080 * scale}" controls>
            <source src="data/video/{file}.MP4" type="video/mp4">
        </video>
    """)
    
def show(config, signal, best_detection_frame=None, 
         predictions=None,
         probabilities=None,
         events_start_time=None, events_end_time=None, 
         events=None, directions=None, views=None,
         from_time=0, till_time=86400,
         save=None,
         plot_true_features=False):

    def formatter(x, y):
        return f'{x // 60:02.0f}:{x % 60:02.0f}'
    
    print(f'{formatter(from_time, None)} - {formatter(till_time, None)}')
    
    signal = signal[from_time * config.sr: till_time * config.sr]
    
    signal_length = len(signal) / config.sr

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

    ax0.xaxis.set_major_formatter(tick.FuncFormatter(formatter))
    ax0.set_xticks(np.arange(x_axis[0], x_axis[-1] + 1, config.window_length))
    ax0.set_xlabel('time [min:sec]')
    
    # show best detection frame from eyedea engine
    if best_detection_frame is not None:
        colors = 'violet'
        mask = (best_detection_frame >= from_time) & (best_detection_frame < till_time)
        
        # color code direction
        if directions is not None:
            colors = ['red' if direction == 'outgoing' else 'green' for direction in directions[mask]]
        
        # color code direction
        if views is not None:
            colors = ['red' if view == 'rear' else 'green' for view in views[mask]]

        ax0.vlines(best_detection_frame[mask], 0, 1, color=colors, linewidth=2.0)
                 
    # show annotations
    if events is not None:
        mask = (events >= from_time) & (events < till_time)
        ax0.vlines(events[mask], 0, 1, color='black', linestyle=':', linewidth=2.0)
        
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
        if signal_length % config.window_length != 0:
            print(f'interval is not divisible by {config.window_length}')
        ax3.xaxis.set_major_formatter(tick.FuncFormatter(formatter))
        ax3.set_xticks(np.arange(x_axis[0], x_axis[-1] + 1, config.window_length))
        from .utils import get_time
        x_axis_time = get_time(signal, config, from_time, till_time)
        #! introduce dummy ending
        predictions = np.append(predictions, 0)
        ax3.step(x_axis_time, predictions, where='post', linewidth=3.0, c='tab:red')
        
        max_output = int(np.max(predictions))
        ax3.hlines(np.arange(1, max_output + 1), x_axis_time[0], x_axis_time[-1], color='k', linestyle='dotted', linewidth=1.0)
        max_output = max(max_output, 1) + 1
        ax3.vlines(x_axis_time, 0, max_output, color='k', linestyle='dotted', linewidth=1, alpha=0.5)
        
        if probabilities is not None:
            x_axis_time = get_time(signal, config, from_time, till_time)
            for x, p in zip(x_axis_time[:-1], probabilities):
                predicted_class = np.argmax(p)
                for i, p_i in zip(range(predicted_class + 1), p):
                    ax3.text(x + 0.5, i + 0.25, f'{i} : {p_i:.4f}', fontsize=13)    
                # ax3.text(x + 0.5, 0.25, f'{np.argmax(p)} : {np.max(p):.4f}', fontsize=13)

        if events is not None:
            events_in_windows = []
            x_axis_time = get_time(signal, config, from_time, till_time)
            for i in range(1, len(x_axis_time)):
                events_in_window = (events >= x_axis_time[i - 1]) & (events < x_axis_time[i])
                events_in_windows.append(events_in_window.sum())
            #! introduce dummy ending
            events_in_windows.append(0)
            ax3.step(x_axis_time, events_in_windows, where='post', linewidth=3.0, linestyle='dotted', c='black')

    # plot signal amplitude
    each = 16
    ax1.plot(signal[::each], alpha=0.5)
    
    features = torch.stft(signal, n_fft=config.n_fft, hop_length=config.hop_length)
    energy = features[..., 0].pow(2)
    energy = energy.sum(0)
    energy = conv(energy)
    energy -= energy.min()
    energy /= energy.max()

    # plot signal energy
    x_axis = np.linspace(0, len(signal[::each]), len(energy))
    ax1_1 = ax1.twinx()
    ax1_1.plot(x_axis, energy, c='black')
    ax1.set_xlabel('number of samples')

    # plot spectrogram
    if plot_true_features:
        transform = create_transformation(config)
        features = transform(signal).squeeze()
    else:
        transform_signal = torchaudio.transforms.MelSpectrogram(sample_rate=config.sr, **get_melkwargs(config))
        transform_power = torchaudio.transforms.AmplitudeToDB(top_db=70)
        features = transform_signal(signal)
        features = transform_power(features)

    ax2.pcolormesh(features)
    ax2.set_xlabel('number of features')
        
    if save is not None and save is not False:
        plt.tight_layout()
        plt.savefig(save, dpi=100)
        
    plt.show()