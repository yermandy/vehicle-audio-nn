from src import *
from sklearn import svm


def append_summary(dict):
    for k, v in dict.items():
        if k == "file":
            dict[k].append("")
        else:
            v = np.array(v).astype(float)
            stats = f"{v.mean():.3f} ± {v.std():.3f}"
            dict[k].append(stats)


def validate_and_save(X, Y, classifier: svm.SVC, files, prefix):
    dict = defaultdict(list)

    for x, y_true, file in zip(X, Y, files):
        y_pred = classifier.predict(x)

        n_pred = y_pred.sum()
        n_true = y_true.sum()

        rvce = abs(n_pred - n_true) / n_true
        error = n_pred - n_true

        dict[f"rvce"].append(f"{rvce:.4f}")
        dict[f"n_events"].append(n_true)
        dict[f"error"].append(error)
        dict[f"file"].append(file)

    append_summary(dict)
    save_dict_txt(
        f"{RESULTS_FOLDER}/{prefix}_svm.txt",
        dict,
    )
    save_dict_csv(
        f"{RESULTS_FOLDER}/{prefix}_svm.csv",
        dict,
    )


def generate_summary_table(uuids, prefix="tst"):
    root_uuid = uuids[0].split("/")[1]
    table = []
    header = []

    for file in files:
        results = pd.read_csv(file, skipfooter=1)
        header = results.columns
        results = results.values
        results = np.atleast_2d(results)
        table.extend(results)
    table = np.array(table).T

    dict = {}
    for i in range(len(header)):
        column_name = header[i]
        column = table[i].tolist()
        dict[column_name] = column

    append_summary(dict)

    save_dict_txt(f"outputs/{root_uuid}/{prefix}_svm.txt", dict)
    save_dict_csv(f"outputs/{root_uuid}/{prefix}_svm.csv", dict)


def get_XY(files: list[str], config: Config, model: nn.Module):
    X = []
    Y = []

    for file in files:
        video = Video(file, config)

        dataset = VehicleDataset(video, part=Part.WHOLE, config=video.config)
        loader = DataLoader(
            dataset,
            batch_size=video.config.batch_size,
            num_workers=video.config.num_workers,
        )

        x = []
        y = []

        device = next(model.parameters()).device

        model.eval()
        with torch.no_grad():
            for tensor, labels in tqdm(loader, leave=True):
                tensor = tensor.to(device)

                x.extend(model.features(tensor).detach().cpu().numpy())
                y.extend(labels[config.svm_predictor.head])

        x = np.array(x)
        y = np.array(y)

        X.append(x)
        Y.append(y)

    return X, Y


def calculate_rvces(classifier, X, Y):
    rvces = []
    for x, y_true in zip(X, Y):
        y_pred = classifier.predict(x)
        rvce = np.abs(y_pred.sum() - y_true.sum()) / y_true.sum()
        rvces.append(rvce)
    return rvces


def calculate_mean_rvce(classifier, X, Y):
    rvces = calculate_rvces(classifier, X, Y)
    return np.mean(rvces)


@hydra.main(config_path="config", config_name="default", version_base="1.2")
def run(config: Config):
    config = Config(config)

    uuid = config.uuid
    split = config.split

    global RESULTS_FOLDER
    RESULTS_FOLDER = f"outputs/{uuid}/results_svm"

    os.makedirs(f"{RESULTS_FOLDER}", exist_ok=True)

    device = get_device(config.cuda)

    model, _ = load_model_locally(uuid, "rvce", device)

    print("extracting X_trn Y_trn")
    trn_files = load_yaml(f"{config.svm_predictor.training_files}/{split}.yaml")
    X_trn, Y_trn = get_XY(trn_files, config, model)

    print("extracting X_val Y_val")
    val_files = load_yaml(f"{config.svm_predictor.validation_files}/{split}.yaml")
    X_val, Y_val = get_XY(val_files, config, model)

    print("extracting X_tst Y_tst")
    tst_files = load_yaml(f"{config.svm_predictor.testing_files}/{split}.yaml")
    X_tst, Y_tst = get_XY(tst_files, config, model)

    X = X_trn if len(X_trn) == 1 else np.concatenate(X_trn)
    Y = Y_trn if len(Y_trn) == 1 else np.concatenate(Y_trn)

    Cs = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]

    lowest_val_rvce = np.inf
    best_C = None

    for C in Cs:
        print(f"training with C={C}")

        classifier = svm.SVC(C=C)
        classifier.fit(X, Y)

        mean_trn_rvce = calculate_mean_rvce(classifier, X_trn, Y_trn)
        print(f"mean trn rvce: {mean_trn_rvce:.4f}")

        mean_val_rvce = calculate_mean_rvce(classifier, X_val, Y_val)
        print(f"mean val rvce: {mean_val_rvce:.4f}")

        if mean_val_rvce < lowest_val_rvce:
            lowest_val_rvce = mean_val_rvce
            best_C = C

    X_trn_and_val = X_trn + X_val
    Y_trn_and_val = Y_trn + Y_val

    X = X_trn_and_val if len(X_trn_and_val) == 1 else np.concatenate(X_trn_and_val)
    Y = Y_trn_and_val if len(Y_trn_and_val) == 1 else np.concatenate(Y_trn_and_val)

    classifier = svm.SVC(C=best_C)
    classifier.fit(X, Y)

    print(f"\nbest C: {best_C}")

    mean_tst_rvce = calculate_mean_rvce(classifier, X_tst, Y_tst)
    print(f"mean tst rvce: {mean_tst_rvce:.4f}")

    validate_and_save(
        X_tst, Y_tst, classifier, tst_files, f"tst_{config.svm_predictor.head}"
    )


def setup_hydra():
    sys.argv.append(f"hydra.output_subdir=svm_predictor_config")
    sys.argv.append(f"hydra/job_logging=disabled")
    sys.argv.append(f"hydra/hydra_logging=none")


@hydra.main(config_path="config", config_name="default", version_base="1.2")
def setup_globals(_config):
    global config
    config = Config(_config)


if __name__ == "__main__":
    setup_hydra()
    setup_globals()

    for head in aslist(config.svm_heads):
        files = []
        for split in aslist(config.svm_splits):
            print(f"split: {split}, head: {head}")
            split_uuid = f"{config.uuid}/{split}"
            sys.argv.append(f"split={split}")
            sys.argv.append(f"svm_predictor.head={head}")
            sys.argv.append(f"root_uuid={config.uuid}")
            sys.argv.append(f"hydra.run.dir=outputs/{split_uuid}")
            sys.argv.append(f"uuid={split_uuid}")

            files.append(f"outputs/{split_uuid}/results_svm/tst_{head}_svm.csv")

            run()

        generate_summary_table(files, f"results_svm/tst/{head}")
