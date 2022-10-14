import argparse
from src import *
from preprocess_data import preprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--videos",
        type=list,
        nargs="+",
        required=True,
        help="name of the video file in data/video folder",
    )
    parser.add_argument("--model", type=str, default="047_october/0", help="model name")
    parser.add_argument(
        "--csv", type=int, choices=[0, 1], default=1, help="csv version"
    )
    parser.add_argument(
        "-i",
        "--image_format",
        type=str,
        choices=["png", "svg"],
        default="svg",
        help="image format",
    )
    args = parser.parse_args()
    args.videos = ["".join(v) for v in args.videos]
    return args


if __name__ == "__main__":

    # parse arguments
    args = parse_args()

    # create results dict
    results_rvces = defaultdict(list)

    # load model and configs
    model, config = load_model_locally(f"{args.model}")

    for video_name in args.videos:

        # get video file name
        video_file_name = get_file_name(video_name)

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

        # get labels
        labels = get_labels(video, from_time, till_time)

        # append file name
        results_rvces["file"].append(video_name)

        # calculate rvce
        for head in predictions.keys():
            rvce_head = calculate_rvce(labels[head], predictions[head])
            results_rvces[head].append(f"{rvce_head:.3f}")

        # plot faults
        for head in predictions.keys():
            visualize_faults(
                predictions[head], labels[head], from_time, till_time, config
            )
            plt.savefig(
                create_subfolders(
                    f"outputs_demo/faults/{video_name}/{head}.{args.image_format}"
                )
            )

    # append summary
    for summary_row, summary_func in zip(
        ["mean", "std", "median"], [np.mean, np.std, np.median]
    ):
        for column_name, rvces in results_rvces.items():
            if column_name == "file":
                results_rvces[column_name].append(summary_row)
            else:
                rvces = np.array(rvces, dtype=np.float32)
                results_rvces[column_name].append(f"{summary_func(rvces):.3f}")

    # save
    save_dict_txt("outputs_demo/results_rvce.txt", results_rvces)
    save_dict_csv("outputs_demo/results_rvce.csv", results_rvces)
