# %%
from src import *
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# %%
m = lambda s: s * 60
h = lambda s: s * 3600


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


def get_XY(files: list[str], config: Config, model: nn.Module, head="n_counts"):
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
        dtype = torch.float16 if config.cuda >= 0 and config.fp16 else torch.float32
        device_type = "cuda" if config.cuda >= 0 else "cpu"

        model.eval()
        with torch.no_grad():
            for tensor, labels in tqdm(loader, leave=True):

                with torch.autocast(device_type=device_type, dtype=dtype):
                    tensor = tensor.to(device)

                x.extend(model.features(tensor).detach().cpu().numpy())
                y.extend(labels[head])

        x = np.array(x)
        y = np.array(y)

        X.append(x)
        Y.append(y)

    return X, Y


def get_XY(files: list[str], config: Config, model: nn.Module, head="n_counts"):
    X = []
    Y = []

    datapool = DataPool(files, config)

    dataset = VehicleDataset(datapool, part=Part.WHOLE, config=video.config)
    loader = DataLoader(
        dataset,
        batch_size=video.config.batch_size,
        num_workers=video.config.num_workers,
    )

    device = next(model.parameters()).device
    dtype = torch.float16 if config.cuda >= 0 and config.fp16 else torch.float32
    device_type = "cuda" if config.cuda >= 0 else "cpu"

    model.eval()
    with torch.no_grad():
        for tensor, labels in tqdm(loader, leave=True):

            with torch.autocast(device_type=device_type, dtype=dtype):
                tensor = tensor.to(device)

            X.extend(model.features(tensor).detach().cpu().numpy())
            Y.extend(labels[head])

    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def calculate_rvce(classifier, x, y_true):
    y_pred = classifier.predict(x)
    rvce = np.abs(y_pred.sum() - y_true.sum()) / y_true.sum()
    return rvce


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

        trn_rvce = calculate_rvce(classifier, X_trn, Y_trn)
        print(f"mean trn rvce: {trn_rvce:.4f}")

        val_rvce = calculate_rvce(classifier, X_val, Y_val)
        print(f"mean val rvce: {val_rvce:.4f}")

        if val_rvce < lowest_val_rvce:
            lowest_val_rvce = val_rvce
            best_C = C

    print("Best C:", best_C)

    return best_C


def find_faults(faults_threshold, y_pred, y_true):
    faults = calculate_cum_errs_video_fault(y_pred, y_true)

    faults_mask = [False] * len(faults)

    for i, f in enumerate(faults):
        if f == faults_threshold:
            for j, f_back in enumerate(faults[:i][::-1]):
                faults_mask[i - j] = True
                if f_back == 0:
                    break
        elif f > faults_threshold:
            faults_mask[i] = True

    return faults_mask


# %%
# def run():

# FILE = ['08-OUT-ALL', 1]
# FILE = ['09-IN-ALL', 1]
# FILE = ['11-IN-ALL', 1]
# FILE = ["05-IN-ALL", 1]
FILE = ["06-IN-ALL", 1]
UUID = "042_large_dataset_1000/0"

FAULTS_THRESHOLD = 5
TRAINING_HOURS = 3
TRAINING_DATASET = f"config/dataset/011_eyedea_cvut.yaml"

model, config = load_model_locally(UUID, model_name="rvce", device="cuda:0")
video = Video(FILE, config)


from_time, till_time = video.get_from_till_time(Part.WHOLE)

# from_time = 0
# till_time = h(5)

predictions, probabilities = validate_video(
    video,
    model,
    from_time=from_time,
    till_time=till_time,
    return_probs=True,
    tqdm=tqdm,
)
labels = get_labels(video, from_time, till_time)

y_true = labels["n_counts"]
y_pred = predictions["n_counts"]

x_axis_time = get_time(config, from_time, till_time)

faults_mask = find_faults(FAULTS_THRESHOLD, y_pred, y_true)

"""
F = np.zeros(len(faults))
F[faults_mask] = max(faults)

plt.figure(figsize=(200, 3))
ax = plt.gca()
ax.margins(0, 0.02)
ax.axline([0, 0], slope=0, ls='--', c='black')
ax.plot(x_axis_time, np.append(faults, 0), c='g', label='video faults')
ax.plot(x_axis_time, np.append(F, 0), c='r', label='faults')
ax.legend(loc='upper left')
set_plt_svg()
# """


X, Y = get_XY([FILE], config, model)

T = int(h(TRAINING_HOURS) // config.window_length)

X_trn, Y_trn = X[:T], Y[:T]
faults_mask = np.asarray(faults_mask)
not_fault_trn = ~faults_mask[:T]

X_trn = X_trn[not_fault_trn]
Y_trn = Y_trn[not_fault_trn]

X_tst, Y_tst = X[T:], Y[T:]
Y_tst_pred = y_pred[T:]

# %%


def get_more_training_data(dataset):

    trn_files = load_yaml(dataset)
    np.random.seed(42)
    np.random.shuffle(trn_files)
    # trn_files = trn_files[:10]

    X, Y = get_XY(trn_files, config, model)
    # X = np.concatenate(X)
    # Y = np.concatenate(Y)

    return X, Y


X_trn_more, Y_trn_more = get_more_training_data(TRAINING_DATASET)

# X_trn = np.concatenate([X_trn, X_trn_more])
# Y_trn = np.concatenate([Y_trn, Y_trn_more])


# %%
def plot_class_distribution(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)

    plt.rcParams["font.size"] = 15

    plt.figure(figsize=(7, 5))
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.bar(unique_labels, counts, align="center")
    plt.tight_layout()


# %%

X_trn_selected = X_trn_more.copy()
Y_trn_selected = Y_trn_more.copy()

mask = (Y_trn_selected != 0) & (Y_trn_selected != 1)
X_trn_selected = X_trn_selected[mask]
Y_trn_selected = Y_trn_selected[mask]

# indices = np.arange(len(X_trn_selected))
# np.random.shuffle(indices)
# indices = indices[:len(X_trn)]

# X_trn_selected = X_trn_selected[indices]
# Y_trn_selected = Y_trn_selected[indices]


# from imblearn.over_sampling import RandomOverSampler
# from imblearn.over_sampling import SMOTE

# sampler = RandomOverSampler(random_state=42)
# sampler = SMOTE(k_neighbors=3)


X_trn_selected = np.concatenate([X_trn, X_trn_selected])
Y_trn_selected = np.concatenate([Y_trn, Y_trn_selected])

# fit predictor and target variable
# X_trn_selected, Y_trn_selected = sampler.fit_resample(X_trn_selected, Y_trn_selected)

# C = find_best_C(X_trn_selected, Y_trn_selected)
C = 50

print("Learning SVM")
# print(np.unique(Y_trn_selected, return_counts=True))
plot_class_distribution(Y_trn_selected)


classifier = make_pipeline(
    StandardScaler(), svm.SVC(C=C, class_weight="balanced", cache_size=5000)
)

classifier.fit(X_trn_selected, Y_trn_selected)

assert Y_tst_pred.shape == Y_tst.shape

tst_rvce_finetuned = calculate_rvce(classifier, X_tst, Y_tst)
tst_rvce_nn = np.abs(Y_tst_pred.sum() - Y_tst.sum()) / Y_tst.sum()

print()
print(f"RVCE NN:  {tst_rvce_nn:.3f}")
print(f"RVCE SVM: {tst_rvce_finetuned:.3f}")

# run()

# %%
