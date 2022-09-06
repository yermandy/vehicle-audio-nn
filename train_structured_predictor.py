from src import *


def load_head_params(model, head_name):
    head = model.heads[head_name]
    w = head.weight.data.detach().cpu().numpy()
    b = head.bias.data.detach().cpu().numpy()
    return w, b


def append_summary(dict):
    for k, v in dict.items():
        if k == "file":
            dict[k].append("")
        else:
            v = np.array(v).astype(float)
            stats = f"{v.mean():.3f} ± {v.std():.3f}"
            dict[k].append(stats)


def validate_and_save(X, Y, w, b, files, uuid, prefix):
    dict = defaultdict(list)

    for x, y_true, file in zip(X, Y, files):
        f = structured_predictor.calc_f(x, w, b)

        _, y_pred = structured_predictor.most_probable_sequence(f)

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
        f"outputs/{uuid}/results_structured_predictor/{prefix}_structured_predictor.txt",
        dict,
    )
    save_dict_csv(
        f"outputs/{uuid}/results_structured_predictor/{prefix}_structured_predictor.csv",
        dict,
    )


def generate_summary_table(uuids, prefix="tst"):
    root_uuid = uuids[0].split("/")[0]
    table = []
    header = []

    for uuid in uuids:
        results = np.genfromtxt(
            f"outputs/{uuid}/results_structured_predictor/{prefix}_structured_predictor.csv",
            delimiter=",",
            skip_footer=1,
            dtype=str,
        )
        header = results[0]
        results = results[1:]
        results = np.atleast_2d(results)
        table.extend(results)
    table = np.array(table).T

    dict = {}
    for i in range(len(header)):
        column_name = header[i]
        column = table[i].tolist()
        dict[column_name] = column

    append_summary(dict)

    save_dict_txt(f"outputs/{root_uuid}/{prefix}_structured_predictor.txt", dict)
    save_dict_csv(f"outputs/{root_uuid}/{prefix}_structured_predictor.csv", dict)


def load_trained_weights(path):
    w = np.load(f"{path}/w.npy")
    b = np.load(f"{path}/b.npy")
    return w, b


def get_XY(files, config, model):
    X = []
    Y = []
    for file in files:
        video = Video(file, config)
        X.append(extract_features(video, model))
        Y.append(extract_labels(video))
    return X, Y


@hydra.main(config_path="config", config_name="default", version_base="1.2")
def run(config: Config):
    config = Config(config)

    uuid = config.uuid
    os.makedirs(f"outputs/{uuid}/results_structured_predictor", exist_ok=True)

    device = get_device(config.cuda)

    model, _ = load_model_locally(uuid, "rvce", device)

    print("extracting X_trn Y_trn")
    trn_files = load_yaml(f"{config.structured_predictor.training_files}/{split}.yaml")
    X_trn, Y_trn = get_XY(trn_files, config, model)

    print("extracting X_val Y_val")
    val_files = load_yaml(
        f"{config.structured_predictor.validation_files}/{split}.yaml"
    )
    X_val, Y_val = get_XY(val_files, config, model)

    print("extracting X_tst Y_tst")
    tst_files = load_yaml(f"{config.structured_predictor.testing_files}/{split}.yaml")
    X_tst, Y_tst = get_XY(tst_files, config, model)

    # set output folder
    config.structured_predictor.outputs_folder = f"outputs/{uuid}/structured_predictor/{config.structured_predictor.head}/{config.structured_predictor.reg}"

    w, b = load_head_params(model, config.structured_predictor.head)

    structured_predictor.learn(
        config.structured_predictor, X_trn, Y_trn, X_val, Y_val, X_tst, Y_tst, w, b
    )

    w, b = load_trained_weights(config.structured_predictor.outputs_folder)

    validate_and_save(
        X_trn,
        Y_trn,
        w,
        b,
        trn_files,
        uuid,
        f"trn_{config.structured_predictor.head}_{config.structured_predictor.reg}",
    )

    validate_and_save(
        X_val,
        Y_val,
        w,
        b,
        val_files,
        uuid,
        f"val_{config.structured_predictor.head}_{config.structured_predictor.reg}",
    )

    validate_and_save(
        X_tst,
        Y_tst,
        w,
        b,
        tst_files,
        uuid,
        f"tst_{config.structured_predictor.head}_{config.structured_predictor.reg}",
    )


def setup_hydra():
    sys.argv.append(r"hydra.run.dir=outputs/${uuid}")
    sys.argv.append(f"hydra.output_subdir=structured_predictor_config")
    sys.argv.append(f"hydra/job_logging=disabled")
    sys.argv.append(f"hydra/hydra_logging=none")


@hydra.main(config_path="config", config_name="default", version_base="1.2")
def setup_globals(_config):
    global config
    config = Config(_config)


if __name__ == "__main__":
    setup_hydra()
    setup_globals()

    uuids = []
    for split in aslist(config.structured_predictor_splits):
        print(f"split: {split}")
        split_uuid = f"{config.uuid}/{split}"
        uuids.append(split_uuid)
        sys.argv.append(f"split={split}")
        sys.argv.append(f"root_uuid={config.uuid}")
        sys.argv.append(f"uuid={split_uuid}")
        sys.argv.append(f"hydra.run.dir=outputs/{split_uuid}")
        run()

    # generate_summary_table(uuids, "trn")
    # generate_summary_table(uuids, "val")
    # generate_summary_table(uuids, "tst")
