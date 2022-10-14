import argparse
from src import *
from preprocess_data import preprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="name of the video file in data/video folder",
    )
    parser.add_argument("--model", type=str, default="047_october/0", help="model name")
    parser.add_argument(
        "--csv", type=int, choices=[0, 1], default=1, help="csv version"
    )
    return parser.parse_args()


if __name__ == "__main__":

    # parse arguments
    args = parse_args()

    # load model and configs
    model, config = load_model_locally(f"{args.model}")

    # get video file name
    video_file_name = get_file_name(args.video)

    # append csv file version
    file = [video_file_name, args.csv]

    # preprocess file
    preprocess([file])

    # preprocess video file to memory
    video = Video(file, config)

    # get video file from and till times
    from_time, till_time = video.get_from_till_time()

    # predict
    predictions, probabilities = validate_video(
        video, model, from_time=from_time, till_time=till_time
    )

    print(predictions)

    # create dict with counts
    results_counts = {k: [np.sum(v)] for k, v in predictions.items()}

    # save
    save_dict_txt("outputs_demo/results_counts.txt", results_counts)
    save_dict_csv("outputs_demo/results_counts.csv", results_counts)

    # create dict with predictions
    results_predictions = defaultdict(list)

    for k, values in predictions.items():
        for value in values:
            results_predictions[k].append(value)

    for i in range(len(values)):
        results_predictions["time_from"].append(i * config.window_length)
        results_predictions["time_till"].append((i + 1) * config.window_length)

    # save
    save_dict_txt("outputs_demo/results_windows.txt", results_predictions)
    save_dict_csv("outputs_demo/results_windows.csv", results_predictions)
