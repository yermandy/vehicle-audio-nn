from src import *


os.makedirs("data/audio_wav", exist_ok=True)
os.makedirs("data/audio_pt", exist_ok=True)
os.makedirs("data/labels", exist_ok=True)
os.makedirs("data/intervals", exist_ok=True)


def extract_audio(file):
    path_pt = find_pt(file)
    path_wav = find_wav(file)

    if path_pt != None:
        print(f"file {path_pt} exists")
        return

    if path_wav != None:
        print(f"file {path_wav} exists")
        return

    path_pt = f"data/audio_pt/{file}.pt"
    path_wav = f"data/audio_wav/{file}.wav"
    # path_mp3 = f"data/audio_wav/{file}.mp3"

    os.makedirs(os.path.dirname(path_wav), exist_ok=True)
    os.makedirs(os.path.dirname(path_pt), exist_ok=True)

    import moviepy.editor as mp

    path_video = find_video(file, True)
    video = mp.VideoFileClip(path_video)
    audio = video.audio
    audio.write_audiofile(path_wav)
    audio.close()
    video.close()

    signal, sr = load_audio_wav(path_wav, return_sr=True)
    torch.save([signal, sr], path_pt)


def optimize(
    views,
    events_start_time,
    events_end_time,
    energy_per_second,
    energy,
    is_rear=True,
    window_len=0.5,
):
    if is_rear:
        mask = views == "rear"
        subset = events_start_time[mask] * energy_per_second
    else:
        mask = views != "rear"
        subset = events_end_time[mask] * energy_per_second

    window_len = window_len * energy_per_second

    delta_best = None
    sum_of_energies_best = 0

    deltas = np.arange(0, 5.1, 0.1)

    for delta in deltas:
        delta = delta * energy_per_second

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
            sum_of_energies += energy[i:j].sum().item()

        if sum_of_energies > sum_of_energies_best:
            sum_of_energies_best = sum_of_energies
            # compensate for half of window length
            delta_best = delta + window_len / 2

    return delta_best / energy_per_second


def extract_labels(file, csv_version):
    path = find_labels(file)
    if path:
        print(f"file {path} exists")
        return
    else:
        path = f"data/labels/{file}.txt"

    csv = load_csv(file, csv_version)
    if csv == []:
        print(f"csv file {file} not found")
        return

    signal, sr = load_audio(file, return_sr=True)
    signal_length = len(signal) // sr

    print("signal_length:", signal_length)

    views = load_views_from_csv(csv, csv_version)
    events_start_time, events_end_time = load_event_time_from_csv(csv, csv_version)

    n_fft = 1024
    hop_length = 128

    energy = torch.stft(signal, n_fft=n_fft, hop_length=hop_length)
    energy = energy[..., 0].pow_(2)
    energy = energy.sum(0)

    energy_per_second = len(energy) / signal_length

    mask = views == "rear"
    if mask.sum() != 0:
        output_rear = optimize(
            views,
            events_start_time,
            events_end_time,
            energy_per_second,
            energy,
            is_rear=True,
        )
        estimated_labels_1 = events_start_time[mask] - output_rear
    else:
        output_rear = 0
        estimated_labels_1 = []
    mask = views != "rear"
    if mask.sum() != 0:
        output_front = optimize(
            views,
            events_start_time,
            events_end_time,
            energy_per_second,
            energy,
            is_rear=False,
        )
        estimated_labels_2 = events_end_time[mask] + output_front
    else:
        output_front = 0
        estimated_labels_2 = []

    print(f"{file}: {output_rear:.2f}, {output_front:.2f}")

    estimated_labels = np.concatenate([estimated_labels_1, estimated_labels_2])
    estimated_labels.sort()
    estimated_labels = np.clip(estimated_labels, 0, signal_length)
    estimated_labels = np.round(estimated_labels, 2)
    estimated_labels.tolist()

    np.savetxt(path, estimated_labels, fmt="%s")


def extract_intevals(file, csv_version, empty_interval_in_s=10):
    path = find_intervals(file)
    if path:
        print(f"file {path} exists")
        return
    else:
        path = f"data/intervals/{file}.txt"

    csv = load_csv(file, csv_version)
    if csv == []:
        print(f"csv file {file} not found")
        return

    CsvColumnID = load_CsvColumnID_version(csv_version)
    events_start_times = load_column(csv, CsvColumnID.START_TIME, csv_version)
    events_end_times = load_column(csv, CsvColumnID.END_TIME, csv_version)

    events_start_times = np.array(
        [time_to_sec(e) for e in events_start_times if e != ""]
    )
    events_end_times = np.array([time_to_sec(e) for e in events_end_times if e != ""])
    # fill space between start and end time
    events = np.linspace(events_start_times, events_end_times, num=100).flatten()
    events = np.sort(events)

    cut_at = next_cut = 0
    intervals = []
    for i in range(1, len(events)):
        diff = events[i] - events[i - 1]
        if diff > empty_interval_in_s:
            cut_at = events[i] - diff / 2
            intervals.append([f"{next_cut:.2f}", f"{cut_at:.2f}"])
            next_cut = cut_at

    signal, sr = load_audio(file, return_sr=True)
    end = len(signal) // sr

    intervals.append([f"{cut_at:.2f}", f"{end:.2f}"])

    np.savetxt(path, intervals, fmt="%s")
    print(path, len(intervals))


def preprocess(dataset):
    for file, csv_version in dataset:
        print("-" * 50)
        print("File: ", file)
        print("\nExtracting audio")
        extract_audio(file)
        print("\nExtracting labels")
        extract_labels(file, csv_version)
        print("\nExtracting intervals")
        extract_intevals(file, csv_version)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "first argument is the path to .yaml list with files"

    # Provide path to .yaml file with list of files

    with open(sys.argv[1], "r") as stream:
        dataset = yaml.safe_load(stream)

    if isinstance(dataset[0], str):
        dataset = [[d, 0] for d in dataset]

    preprocess(dataset)
