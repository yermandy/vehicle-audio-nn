from src import *


def combine_short_to_long(folder_root, out_file):
    files = sorted(glob(f"data/audio_pt/{folder_root}/*"))

    combined = []

    for file in files:
        if "ALL" in file:
            continue
        signal, sr = load_audio_tensor(file, True)

        print(signal.shape)

        combined.append(signal)

    combined = torch.cat(combined)

    print(combined.shape)
    torch.save([combined, sr], f"data/audio_pt/{folder_root}/{out_file}")


if __name__ == "__main__":
    folder_root = "06-IN"
    out_file = "06-IN-ALL.pt"
    combine_short_to_long(folder_root, out_file)
