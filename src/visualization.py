import torch
import torchaudio
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as tick

import warnings
import os

from .transformation import create_transformation
from .utils import crop_signal, get_n_hops, get_signal_length, create_samples

warnings.filterwarnings("ignore")


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def time_formatter(s, *args):
    s = int(s)
    m = s // 60
    h = m // 60
    return f"{h:02d}:{m % 60:02d}:{s % 60:02d}"


def get_confusion_matrix(labels, predictions):
    labels = np.array(labels).flatten()
    predictions = np.array(predictions).flatten()
    conf_matrix = confusion_matrix(labels, predictions)
    return conf_matrix


def plot_confusion_matrix(labels, predictions):
    print(
        f"pred: {predictions.sum()} \t true: {labels.sum()} \t rvce: {abs(predictions.sum() - labels.sum()) / labels.sum():.3f}"
    )

    conf_matrix = get_confusion_matrix(labels, predictions)

    conf_matrix_norm = (conf_matrix.T / conf_matrix.sum(1)).T
    conf_matrix_norm[np.isnan(conf_matrix_norm)] = 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay(conf_matrix).plot(ax=axes[0])
    ConfusionMatrixDisplay(conf_matrix_norm).plot(ax=axes[1])


def get_melkwargs(params):
    return {
        "n_fft": params.n_fft,
        "n_mels": params.n_mels,
        "hop_length": params.hop_length,
    }


def conv(array):
    array = array.view(1, 1, -1)
    kernel = torch.ones(1, 1, 101)
    array = torch.nn.functional.conv1d(array, kernel, padding="same")
    array = array.squeeze()
    return array


def show_video(file, scale=0.3):
    from IPython.display import HTML

    return HTML(
        f"""
        <video width="{1920 * scale}" height="{1080 * scale}" controls>
            <source src="data/video/{file}.MP4" type="video/mp4">
        </video>
    """
    )


def plot_class_distribution(labels, normalized=False):
    unique_labels, counts = np.unique(labels, return_counts=True)
    if normalized:
        counts = counts / counts.sum()
        plt.ylabel("Distribution of samples")
    else:
        plt.ylabel("Number of samples")

    plt.xlabel("Class")
    plt.bar(unique_labels, counts, align="center")
    plt.tight_layout()


def show(
    config,
    signal,
    best_detection_frame=None,
    predictions=None,
    probabilities=None,
    events_start_time=None,
    events_end_time=None,
    events=None,
    directions=None,
    views=None,
    from_time=0,
    till_time=86400,
    save=None,
    plot_true_features=False,
    width_multiplier=4,
):

    signal_length = get_signal_length(signal, config)
    if till_time > signal_length:
        print("till_time > signal_length")
        till_time = signal_length

    def formatter(x, y):
        return f"{x // 60:02.0f}:{x % 60:02.0f}"

    signal = crop_signal(signal, config.sr, from_time, till_time)
    signal_length = get_signal_length(signal, config)

    print(f"{formatter(from_time, None)} - {formatter(till_time, None)}")

    nrows = 3
    if predictions is not None:
        nrows += 1
    width = (till_time - from_time) // width_multiplier
    height = 4 * nrows
    fig, axes = plt.subplots(nrows=nrows, figsize=(width, height))

    if predictions is not None:
        ax0, ax3, ax1, ax2 = axes
    else:
        ax0, ax1, ax2 = axes

    for ax in axes:
        ax.margins(0, 0.02)

    x_axis = np.arange(from_time, till_time + 1)
    ax0.plot(x_axis, np.zeros(len(x_axis)), marker="o", markersize=3, color="black")
    ax0.get_yaxis().set_visible(False)

    ax0.xaxis.set_major_formatter(tick.FuncFormatter(formatter))
    ax0.set_xticks(np.arange(x_axis[0], x_axis[-1] + 1, config.window_length))
    # ax0.set_xlabel('time [min:sec]')

    # show windows
    # x_axis = np.linspace(from_time, till_time, get_n_hops(config, from_time, till_time) + 1)
    # ax0.vlines(x_axis, 0, 1, color='tab:red', linestyle='--', linewidth=2.0)

    # show best detection frame from eyedea engine
    if best_detection_frame is not None:
        colors = "violet"
        mask = (best_detection_frame >= from_time) & (best_detection_frame < till_time)

        # color code direction
        if directions is not None:
            colors = [
                "red" if direction == "outgoing" else "green"
                for direction in directions[mask]
            ]

        # color code direction
        if views is not None:
            colors = ["red" if view == "rear" else "green" for view in views[mask]]

        ax0.vlines(best_detection_frame[mask], 0, 1, color=colors, linewidth=2.0)

    # show annotations
    if events is not None and len(events) > 0:
        mask = (events >= from_time) & (events < till_time)
        ax0.vlines(events[mask], 0, 1, color="black", linestyle=":", linewidth=2.0)

    # show start and end time of events
    if (
        events_start_time is not None
        and len(events_start_time) > 0
        and events_end_time is not None
        and len(events_end_time) > 0
    ):
        colors = "violet"

        # color code direction
        if directions is not None:
            colors = [
                "red" if direction == "outgoing" else "green"
                for direction in directions
            ]

        # color code direction
        if views is not None:
            colors = ["red" if view == "rear" else "green" for view in views]

        for event_start_time, event_end_time, color in zip(
            events_start_time, events_end_time, colors
        ):
            if (
                from_time <= event_start_time <= till_time
                and from_time <= event_end_time <= till_time
            ):
                ax0.fill_between(
                    [event_start_time, event_end_time],
                    [0],
                    [1],
                    color=color,
                    alpha=0.25,
                )

    # plot predictions
    if predictions is not None:
        x_axis = np.arange(from_time, till_time + 1)
        if signal_length % config.window_length != 0:
            print(f"interval is not divisible by {config.window_length}")
        ax3.xaxis.set_major_formatter(tick.FuncFormatter(formatter))
        ax3.set_xticks(np.arange(x_axis[0], x_axis[-1] + 1, config.window_length))
        from .utils import get_time

        x_axis_time = get_time(config, from_time, till_time)
        #! introduce dummy ending
        predictions = np.append(predictions, 0)
        ax3.step(x_axis_time, predictions, where="post", linewidth=3.0, c="tab:green")

        max_output = int(np.max(predictions))
        ax3.hlines(
            np.arange(1, max_output + 1),
            x_axis_time[0],
            x_axis_time[-1],
            color="k",
            linestyle="dotted",
            linewidth=1.0,
        )
        max_output = max(max_output, 1) + 1
        ax3.vlines(
            x_axis_time,
            0,
            max_output,
            color="k",
            linestyle="dotted",
            linewidth=1,
            alpha=0.5,
        )

        if probabilities is not None:
            x_axis_time = get_time(config, from_time, till_time)
            for x, p in zip(x_axis_time[:-1], probabilities):

                # predicted_class = np.argmax(p)
                # for i, p_i in zip(range(predicted_class + 1), p):
                #     ax3.text(x + 0.5, i + 0.25, f'{i} : {p_i:.4f}', fontsize=13)
                # ax3.text(x + 0.5, 0.25, f'{np.argmax(p)} : {np.max(p):.4f}', fontsize=13)

                for i, p_i in zip(range(max_output), p):
                    ax3.text(x + 1, i + 0.25, f"{i} : {p_i:.4f}", fontsize=13)

        if events is not None:
            events_in_windows = []
            x_axis_time = get_time(config, from_time, till_time)
            for i in range(1, len(x_axis_time)):
                events_in_window = (events >= x_axis_time[i - 1]) & (
                    events < x_axis_time[i]
                )
                events_in_windows.append(events_in_window.sum())
            #! introduce dummy ending
            events_in_windows.append(0)
            ax3.step(
                x_axis_time,
                events_in_windows,
                where="post",
                linewidth=3.0,
                linestyle="dotted",
                c="black",
            )

        ax3.yaxis.set_major_locator(tick.MaxNLocator(integer=True))
        # ax3.set_xlabel('time [min:sec]')
        ax3.get_yaxis().set_visible(False)

    # plot signal amplitude
    each = 16
    smaller_signal = signal[::each]
    ax1.plot(smaller_signal, alpha=0.5)

    features = torch.stft(signal, n_fft=config.n_fft, hop_length=config.hop_length)
    energy = features[..., 0].pow(2)
    energy = energy.sum(0)
    energy = conv(energy)
    energy -= energy.min()
    energy /= energy.max()

    # plot signal energy
    x_axis = np.linspace(0, len(smaller_signal), len(energy))
    ax1_1 = ax1.twinx()
    ax1_1.plot(x_axis, energy, c="black")
    # ax1.set_xlabel('number of samples')

    # plot windows
    x_axis = np.linspace(
        0, len(smaller_signal), get_n_hops(config, from_time, till_time) + 1
    )
    ax1.vlines(x_axis, -1, 1, color="k", linestyle="--", linewidth=2.0, alpha=0.5)
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax1_1.get_yaxis().set_visible(False)

    # plot spectrogram
    if plot_true_features:
        transform = create_transformation(config)
        samples = create_samples(config, signal, from_time, till_time)
        features = [transform(sample).squeeze() for sample in samples]
        features = torch.hstack(features)
    else:
        transform_signal = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr, **get_melkwargs(config)
        )
        transform_power = torchaudio.transforms.AmplitudeToDB(top_db=70)
        features = transform_signal(signal)
        features = transform_power(features)

    ax2.pcolormesh(features)
    ax2.set_xlabel("number of features")

    # plot windows
    x_axis = np.linspace(
        0, features.shape[1], get_n_hops(config, from_time, till_time) + 1
    )
    ax2.vlines(x_axis, 0, features.shape[0], color="k", linestyle="--", linewidth=2.0)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    if save is not None and save is not False:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save, dpi=100)
        plt.close()
    else:
        plt.show()


def set_plt_svg():
    import matplotlib_inline.backend_inline

    matplotlib_inline.backend_inline.set_matplotlib_formats("svg")


def set_plt_png():
    import matplotlib_inline.backend_inline

    matplotlib_inline.backend_inline.set_matplotlib_formats("png")
