from operator import delitem
import yaml
import librosa
import argparse
import numpy as np

if __name__ == "__main__":

    # file = 'config/dataset/dataset_26.11.2021.yaml'
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    file = args.dataset

    with open(file, "r") as stream:
        files = yaml.safe_load(stream)

    total_duration = []
    total_n_events = []

    rows = ["file name, duration in seconds, number of video events"]
    for file in files:
        duration = round(librosa.get_duration(filename=f"data/audio/{file}.MP4.wav"))
        n_events = len(np.loadtxt(f"data/labels/{file}.MP4.txt"))
        rows.append(f"{file}, {duration}, {n_events}")

        total_duration.append(duration)
        total_n_events.append(n_events)

    rows.append(
        f"total, {np.mean(total_duration):.2f} ± {np.std(total_duration):.2f}, {np.mean(total_n_events):.2f} ± {np.std(total_n_events):.2f}"
    )
    rows = np.array(rows)

    np.savetxt("file_duration_video_events.csv", rows, fmt="%s", delimiter=",")
