# %%
import pickle
from src import *
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# %%


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


def find_best_C(X, Y):
    X_trn, X_val, Y_trn, Y_val = train_test_split(X, Y, test_size=0.5, random_state=42)

    # Cs = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    Cs = [1, 5, 10, 50, 100, 500, 1000]

    lowest_val_rvce = np.inf
    best_C = None

    for C in Cs:
        print(f"\ntraining with C={C}")

        # classifier = svm.SVC(C=C)
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        classifier = make_pipeline(StandardScaler(), svm.SVC(C=C))

        classifier.fit(X, Y)

        trn_rvce = calculate_rvce_of_svm_classifier(classifier, X_trn, Y_trn)
        print(f"mean trn rvce: {trn_rvce:.4f}")

        val_rvce = calculate_rvce_of_svm_classifier(classifier, X_val, Y_val)
        print(f"mean val rvce: {val_rvce:.4f}")

        if val_rvce < lowest_val_rvce:
            lowest_val_rvce = val_rvce
            best_C = C

    print("Best C:", best_C)

    return best_C


def get_more_training_data(dataset, config, model, head):
    trn_files = load_yaml(dataset)
    np.random.seed(42)
    np.random.shuffle(trn_files)

    X, Y = get_XY(trn_files, config, model, head)

    return X, Y


def run(
    uuid,
    file,
    head,
    faults_threshold,
    training_hours,
    X_trn_more,
    Y_trn_more,
    path=None,
):
    print(f"uuid: {uuid}")
    print(f"file: {file}")
    print(f"head: {head}")
    print(f"faults_threshold: {faults_threshold}")
    print(f"training_hours: {training_hours}")

    model, config = load_model_locally(uuid, model_name="rvce", device="cuda:0")
    video = Video(file, config)

    from_time, till_time = video.get_from_till_time(Part.WHOLE)

    predictions, probabilities = validate_video(
        video,
        model,
        from_time=from_time,
        till_time=till_time,
        tqdm=tqdm,
    )
    labels = get_labels(video, from_time, till_time)

    y_true = labels[head]
    y_pred = predictions[head]

    X, Y = get_XY([file], config, model, head)

    X_trn, Y_trn, X_tst, Y_tst, Y_tst_pred = remove_faults(
        config, training_hours, faults_threshold, y_pred, y_true, X, Y
    )

    X_trn_NN, Y_trn_NN = preprocess_data(X_trn_more, Y_trn_more)

    X_trn_SVM = np.concatenate([X_trn, X_trn_NN])
    Y_trn_SVM = np.concatenate([Y_trn, Y_trn_NN])

    # C = find_best_C(X_trn_NN, Y_trn_NN)
    C = 50

    print("Learning SVM")
    plot_class_distribution(Y_trn_NN)
    plt.show()

    plot_class_distribution(Y_trn_SVM)
    plt.show()

    classifier = make_pipeline(
        StandardScaler(), svm.SVC(C=C, class_weight="balanced", cache_size=5000)
    )

    classifier.fit(X_trn_SVM, Y_trn_SVM)

    assert Y_tst_pred.shape == Y_tst.shape

    tst_rvce_finetuned = calculate_rvce_of_svm_classifier(classifier, X_tst, Y_tst)
    tst_rvce_nn = calculate_rvce(Y_tst, Y_tst_pred)

    print()
    print(f"RVCE NN:  {tst_rvce_nn:.3f}")
    print(f"RVCE SVM: {tst_rvce_finetuned:.3f}")

    file_name = file if isinstance(file, str) else file[0]

    if path == None:
        path = f"outputs/{uuid}/svm/{file_name}/{head}/model.pkl"
    create_subfolders(path)

    # save
    with open(path, "wb") as f:
        pickle.dump(classifier, f)


# %%
if __name__ == "__main__":
    # parameters
    UUIDS = ["042_large_dataset_1000/0", "047_october/0"]
    UUID = UUIDS[1]
    HEADS = ["n_counts", "n_incoming", "n_outgoing", "n_CAR", "n_NOT_CAR"]
    HEAD = HEADS[0]
    FILES = [
        ["08-OUT-ALL", 1],
        ["09-IN-ALL", 1],
        ["11-IN-ALL", 1],
        ["05-IN-ALL", 1],
        ["06-IN-ALL", 1],
    ]
    FAULTS_THRESHOLD = 5
    TRAINING_HOURS = 6
    TRAINING_DATASET = f"config/dataset/011_eyedea_cvut.yaml"

    # %%

    model, config = load_model_locally(UUID, model_name="rvce", device="cuda:0")

    X_trn_more, Y_trn_more = get_more_training_data(
        TRAINING_DATASET, config, model, HEAD
    )

    # %%

    model, config = load_model_locally(UUID, model_name="rvce", device="cuda:0")

    for head in HEADS:
        X_trn_more, Y_trn_more = get_more_training_data(
            TRAINING_DATASET, config, model, head
        )

        for file in FILES:
            run(
                UUID,
                file,
                head,
                FAULTS_THRESHOLD,
                TRAINING_HOURS,
                X_trn_more,
                Y_trn_more,
            )
# %%
