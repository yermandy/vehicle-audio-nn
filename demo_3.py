import os

os.environ["NO_CACHE"] = "1"

# %%
import argparse
from src import *
from preprocess_data import preprocess
from finetune_long_sequences import run

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        "-v",
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
    parser.add_argument("--training_hours", type=float, default=6, help="training time")
    parser.add_argument(
        "--faults_threshold", type=float, default=6, help="training time"
    )
    parser.add_argument(
        "-i",
        "--image_format",
        type=str,
        choices=["png", "svg"],
        default="svg",
        help="image format",
    )
    return parser.parse_args()


def validate(predictions, labels, head, path, training_hours):
    T = int(h(training_hours) // config.window_length)

    y_pred_nn = predictions[head]
    y_true = labels[head]

    y_pred_nn = y_pred_nn[T:]
    y_true = y_true[T:]

    with open(path, "rb") as f:
        classifier = pickle.load(f)

    X, Y = get_XY([file], config, model, head)

    X_tst = X[T:]
    y_pred_svm = classifier.predict(X_tst)

    assert len(y_pred_svm) == len(y_pred_nn)

    return y_pred_nn, y_pred_svm, y_true


# %%
if __name__ == "__main__":

    # parse arguments
    args = parse_args()

    # load model and configs
    model, config = load_model_locally(f"{args.model}", device=args.device)

    # get video file name
    video_file_name = get_file_name(args.video)

    # append csv file version
    file = [video_file_name, args.csv]

    # preprocess file
    preprocess([file])

    # load video
    video = Video(file, config)

    from_time, till_time = video.get_from_till_time(Part.WHOLE)

    # validate video
    predictions, probabilities = validate_video(
        video,
        model,
        from_time=from_time,
        till_time=till_time,
        return_probs=True,
        tqdm=tqdm,
    )

    # load labels
    labels = get_labels(video, from_time, till_time)

    # select training files
    files = config.training_files + config.validation_files + config.testing_files

    # load more features
    X_trn_more, Y_trn_more_heads = get_XY_heads(files, config, model)

    X, Y_heads = get_XY_heads([file], config, model)

    for head in config.heads.keys():
        Y_trn_more = Y_trn_more_heads[head]
        Y = Y_heads[head]

        X_trn, Y_trn, X_tst, y_true, y_pred_nn = split_and_remove_faults(
            config,
            args.training_hours,
            args.faults_threshold,
            predictions[head],
            labels[head],
            X,
            Y,
        )

        X_trn_NN, Y_trn_NN = preprocess_data(X_trn_more, Y_trn_more)

        X_trn_SVM = np.concatenate([X_trn, X_trn_NN])
        Y_trn_SVM = np.concatenate([Y_trn, Y_trn_NN])

        # C = find_best_C(X_trn_NN, Y_trn_NN)
        C = 50

        # instantiate model
        classifier = make_pipeline(
            StandardScaler(), svm.SVC(C=C, class_weight="balanced", cache_size=5000)
        )

        # fit svm model
        classifier.fit(X_trn_SVM, Y_trn_SVM)

        # save
        path = f"outputs_demo/{video_file_name}/svm/models/{head}/model.pkl"
        with open(create_subfolders(path), "wb") as f:
            pickle.dump(classifier, f)

        y_pred_svm = classifier.predict(X_tst)

        rvce_nn = calculate_rvce(y_true, y_pred_nn)
        rvce_svm = calculate_rvce(y_true, y_pred_svm)

        plot_confusion_matrix(y_true, y_pred_nn)
        plt.savefig(
            create_subfolders(
                f"outputs_demo/{video_file_name}/svm/confusion/nn_{head}.{args.image_format}"
            )
        )

        plot_confusion_matrix(y_true, y_pred_svm)
        plt.savefig(
            create_subfolders(
                f"outputs_demo/{video_file_name}/svm/confusion/svm_{head}.{args.image_format}"
            )
        )

        print(f"head:     {head}")
        print(f"RVCE NN:  {rvce_nn:.3f}")
        print(f"RVCE SVM: {rvce_svm:.3f}")

        visualize_faults(y_pred_nn, y_true, h(args.training_hours), till_time, config)
        plt.savefig(
            create_subfolders(
                f"outputs_demo/{video_file_name}/svm/faults/nn_{head}.{args.image_format}"
            )
        )

        visualize_faults(y_pred_svm, y_true, h(args.training_hours), till_time, config)
        plt.savefig(
            create_subfolders(
                f"outputs_demo/{video_file_name}/svm/faults/svm_{head}.{args.image_format}"
            )
        )

        # %%
