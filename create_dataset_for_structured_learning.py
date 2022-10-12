from src import *
import argparse


def load_true_labels(video, from_time, till_time):
    video.config.set_window_length(3)
    video.config.set_nn_hop_length(3)
    _, labels = create_dataset_from_video(
        video, from_time=from_time, till_time=till_time
    )
    labels = np.array(labels["n_counts"])
    print("labels length:", len(labels), " | counts:", labels.sum())
    return labels


def load_head_params(model):
    head = model.heads["n_counts"]
    w = head.weight.data
    b = head.bias.data
    return w, b


def load_features(video, from_time, till_time):
    video.config.set_window_length(6)
    video.config.set_nn_hop_length(3)
    samples, labels = create_dataset_from_video(
        video, from_time=from_time, till_time=till_time
    )
    labels = np.array(labels["n_counts"])

    transformation = create_transformation(config)

    n = len(samples)
    features = []

    with torch.no_grad():
        for i in range(n - 1):
            sample = samples[i]
            sample = transformation(sample).unsqueeze(0).to(device)
            features_sample = model.features(sample).squeeze().detach().cpu().numpy()

            features.append(features_sample)
            # print(outputs.softmax(0).detach().numpy())

    features = np.array(features)

    print("features shape:", features.shape)

    return features


def minutes(minutes):
    return minutes * 60


def generate_shuffled_set(portion=0.6):
    together = config.training_files + config.validation_files
    np.random.shuffle(together)
    split = int(len(together) * portion)
    return together[:split], together[split:]


def create_dataset(
    part, n_minutes, n_samples, is_shuffled=False, whole_file=False, split_number=0
):
    # save_params(model)
    # w, b = load_head_params(model)

    labels_all = []
    features_all = []

    np.random.seed(0)

    if part == "tst":
        files = config.testing_files
    elif is_shuffled:
        training_files, validation_files = generate_shuffled_set()
        files = training_files if part == "trn" else validation_files
    else:
        files = config.training_files if part == "trn" else config.validation_files

    for i, file in enumerate(files):
        print(i + 1, file)

        video = Video(file, config)
        duration = video.get_from_till_time()[1]
        # print('duration:', duration)

        for i in range(n_samples):
            if whole_file:
                from_time = None
                till_time = None
            else:
                samples_length = minutes(n_minutes)
                from_time = np.random.randint(0, duration - samples_length)
                till_time = from_time + samples_length

            # print('from_time:', from_time)
            labels = load_true_labels(video, from_time, till_time)
            features = load_features(video, from_time, till_time)

            labels_all.append(labels)
            features_all.append(features)

        # scores = w @ features[::2].T + b.reshape(-1, 1)
        # y_pred = scores.argmax(0)
        # c_pred = y_pred.sum()
        # c_true = labels.sum()
        # rvce = abs(c_pred - c_true) / c_true
        # rvces.append(rvce)

    # print(np.mean(rvces))

    folder = f"{root_folder}/"

    if part == "tst":
        folder += "tst/"
    elif part == "trn":
        folder += "trn/"
    else:
        folder += "val/"

    if split_number is not None:
        folder += f"split_{split_number}/"

    if is_shuffled:
        folder += "shuffled/"

    if whole_file:
        folder += "whole_file/"
    else:
        folder += f"{n_minutes}_minutes/"
        folder += f"{n_samples}_samples/"

    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/y.pickle", "wb") as f:
        pickle.dump(labels_all, f)

    with open(f"{folder}/features.pickle", "wb") as f:
        pickle.dump(features_all, f)


def save_params(split_number, model):
    w, b = load_head_params(model)
    folder = f"{root_folder}/params/split_{split_number}"
    os.makedirs(folder, exist_ok=True)
    np.save(f"{folder}/w.npy", w.detach().cpu().numpy())
    np.save(f"{folder}/b.npy", b.detach().cpu().numpy())
    return w, b


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_number", type=int, default=0, help="split number")
    parser.add_argument(
        "--part", type=str, choices=["tst", "val", "trn"], default="tst", help="part"
    )
    return parser.parse_args()


if __name__ == "__main__":

    root_folder = "outputs/000_structured_rvce/036"

    args = parse_args()

    uuid = (
        f"036_aligned_resized_128_audio_image_augmentation_bs_256/{args.split_number}"
    )

    model_name = "rvce"
    device = "cuda:0"
    # device = 'cpu'

    model, config = load_model_locally(uuid, model_name=model_name, device=device)

    save_params(args.split_number, model)

    create_dataset(
        part=args.part,
        n_minutes=None,
        n_samples=1,
        is_shuffled=True,
        whole_file=True,
        split_number=args.split_number,
    )
