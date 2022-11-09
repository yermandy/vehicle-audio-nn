import numpy as np

from .visualization import *
from .utils import *


def calculate_cum_errs_video_fault(y_pred, y_true):
    cum_errs = []
    cum_err = 0
    for yp, yt in zip(y_pred, y_true):
        if yp == yt and yp != 0:
            cum_err = 0
        elif yp != yt and yt != 0:
            cum_err = 0
        else:
            cum_err += yp - yt
        cum_errs.append(cum_err)
    return np.array(cum_errs)


def calculate_cum_errs_audio_fault(y_pred, y_true):
    cum_errs = []
    cum_err = 0
    for yp, yt in zip(y_pred, y_true):
        if yp == yt and yp != 0:
            cum_err = 0
        elif yp != yt and yp != 0:
            cum_err = 0
        else:
            cum_err += yt - yp
        cum_errs.append(cum_err)
    return np.array(cum_errs)


def visualize_faults(y_pred, y_true, from_time, till_time, config):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    resolution = 300
    n_last_elements = int(m(30) / config.window_length)
    tick_frequency = m(5)

    time_axis = np.arange(from_time, till_time + 1)

    # fig, axes = plt.subplots(3, figsize=((till_time - from_time) / resolution, 7))
    fig, axes = plt.subplots(5, figsize=((till_time - from_time) / resolution, 12))

    for ax in axes:
        ax.margins(0, 0.02)
        ax.xaxis.set_major_formatter(tick.FuncFormatter(time_formatter))
        ax.set_xticks(np.arange(time_axis[0], time_axis[-1] + 1, tick_frequency))

    x_axis_time = get_time(config, from_time, till_time)

    # plot true
    axes[0].plot(x_axis_time, np.append(y_true, 0), c="g", label="video")

    # plot predicted
    axes[0].plot(x_axis_time, np.append(y_pred, 0), c="b", label="audio", ls=":")
    axes[0].legend(loc="upper left")

    # plot difference
    diff = y_pred - y_true

    windowed_counting_error = []
    for till_index in range(len(diff)):
        from_index = max(0, till_index - n_last_elements)
        windowed_counting_error.append(
            y_pred[from_index:till_index].sum() - y_true[from_index:till_index].sum()
        )

    axes[1].plot(x_axis_time, np.append(diff, 0), c="r", label="audio - video")
    axes[1].legend(loc="upper left")

    axes[2].axline([x_axis_time[0], 0], slope=0, ls="--", c="black")
    axes[2].plot(
        x_axis_time, np.append(windowed_counting_error, 0), c="r", label="30 min error"
    )
    axes[2].legend(loc="upper left")

    # calculate rvces
    rvces = []
    for till_index in range(len(diff)):
        from_index = max(0, till_index - n_last_elements)
        y_pred_sum = y_pred[:till_index].sum()
        y_true_sum = y_true[:till_index].sum()
        rvces.append(abs(y_pred_sum - y_true_sum) / y_true_sum)

    axes[3].axline([x_axis_time[0], 0], slope=0, ls="--", c="black")
    axes[3].axline(
        [x_axis_time[0], rvces[-1]],
        slope=0,
        ls="--",
        c="red",
        label=f"last={rvces[-1]:.3f}",
    )
    axes[3].plot(x_axis_time, np.append(rvces, 0), c="r", label="rvce")
    axes[3].legend(loc="upper left")

    axes[4].axline([x_axis_time[0], 0], slope=0, ls="--", c="black")
    axes[4].plot(
        x_axis_time,
        np.append(calculate_cum_errs_video_fault(y_pred, y_true), 0),
        c="g",
        label="video faults",
    )
    axes[4].plot(
        x_axis_time,
        np.append(calculate_cum_errs_audio_fault(y_pred, y_true), 0),
        c="b",
        label="audio faults",
    )
    axes[4].legend(loc="upper left")

    plt.tight_layout()
    set_plt_svg()


def find_faults(faults_threshold, y_pred, y_true):
    faults = calculate_cum_errs_video_fault(y_pred, y_true)

    faults_mask = [False] * len(faults)

    for i, f in enumerate(faults):
        if f == faults_threshold:
            for j, f_back in enumerate(faults[:i][::-1]):
                faults_mask[i - j] = True
                if f_back == 0:
                    break
        elif f > faults_threshold:
            faults_mask[i] = True

    return faults_mask


def remove_faults(config, traing_hours, faults_threshold, y_pred, y_true, X, Y):
    # find training part
    T = int(h(traing_hours) // config.window_length)

    # copy all data
    X_trn = X[:T].copy()
    Y_trn = Y[:T].copy()
    X_tst = X[T:].copy()
    Y_tst = Y[T:].copy()
    Y_tst_pred = y_pred[T:].copy()

    # find fault intervals
    faults_mask = find_faults(faults_threshold, y_pred, y_true)
    faults_mask = np.asarray(faults_mask)
    not_fault_trn = ~faults_mask[:T]

    # remove fault intervals from training data
    X_trn = X_trn[not_fault_trn]
    Y_trn = Y_trn[not_fault_trn]

    return X_trn, Y_trn, X_tst, Y_tst, Y_tst_pred


def preprocess_data(X_trn_more, Y_trn_more):

    X_trn_NN = X_trn_more.copy()
    Y_trn_NN = Y_trn_more.copy()

    mask = (Y_trn_NN != 0) & (Y_trn_NN != 1)
    X_trn_NN = X_trn_NN[mask]
    Y_trn_NN = Y_trn_NN[mask]

    # indices = np.arange(len(X_trn_NN))
    # np.random.shuffle(indices)
    # indices = indices[:len(X_trn)]

    # X_trn_NN = X_trn_NN[indices]
    # Y_trn_NN = Y_trn_NN[indices]

    return X_trn_NN, Y_trn_NN
