import argparse
from src import *
from preprocess_data import preprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        required=True,
        help="name of the video file in data/video folder",
    )
    parser.add_argument(
        "--model", "-m", type=str, default="047_october/0", help="model name"
    )
    parser.add_argument(
        "--csv", type=int, choices=[0, 1], default=1, help="csv version"
    )
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "-i",
        "--image_format",
        type=str,
        choices=["png", "svg"],
        default="svg",
        help="image format",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # parse arguments
    args = parse_args()

    # create results dict
    results_rvces = defaultdict(list)

    # load model and configs
    model, config = load_model_locally(f"{args.model}", device=args.device)

    video_name = args.video

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

    # calculate rvce
    for head in predictions.keys():
        rvce_head = calculate_rvce(labels[head], predictions[head])
        results_rvces[head].append(f"{rvce_head:.3f}")

    # plot faults
    for head in predictions.keys():
        visualize_faults(predictions[head], labels[head], from_time, till_time, config)
        plt.savefig(
            create_subfolders(
                f"outputs_demo/{video_name}/faults/{head}.{args.image_format}"
            )
        )

    # plot confusion matrices
    for head in predictions.keys():
        plot_confusion_matrix(labels[head], predictions[head])
        plt.savefig(
            create_subfolders(
                f"outputs_demo/{video_name}/confusion/{head}.{args.image_format}"
            )
        )

    # save
    save_dict_txt(f"outputs_demo/{video_name}/results_rvce.txt", results_rvces)
    save_dict_csv(f"outputs_demo/{video_name}/results_rvce.csv", results_rvces)
