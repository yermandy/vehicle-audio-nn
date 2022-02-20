# %%
import os
os.chdir('..')

from sklearn.svm import LinearSVC

from src import *
import yaml
from tabulate import tabulate

# %%
with open(f'config/dataset/dataset_26.11.2021.yaml', 'r') as stream:
    dataset = yaml.safe_load(stream)
datapool = DataPool(dataset, 6.0, 0.75, tqdm=tqdm)

# %%

fold = 0
model, params = load_model_locally(f'cross_validation_25/{fold}', model_name='rvce')
transformation = create_transformation(params)

folds = 25
trn_folds = [f'config/training_files/cross_validation_28.01.2021/{i}.yaml' for i in range(19, folds)]
tst_folds = [f'config/testing_files/cross_validation_28.01.2021/{i}.yaml' for i in range(19, folds)]

# %%

table = []

for fold, (trn_fold, tst_fold) in enumerate(zip(trn_folds, tst_folds)):

    X_trn = []
    y_trn = []

    X_tst = []
    y_tst = []

    with open(trn_fold, 'r') as stream:
        trn_files = yaml.safe_load(stream)
    with open(tst_fold, 'r') as stream:
        tst_files = yaml.safe_load(stream)

    for file in tst_files:
        video = datapool[file]
        events = video.get_events(True)
        trn_samples_positive, _ = create_dataset_sequentially(
            video.signal, params.sr, video.events, 
            from_time=video.trn_from_time, 
            till_time=video.trn_till_time, 
            window_length=params.window_length
        )

        X_trn.extend(trn_samples_positive)
        y_trn.extend([1] * len(trn_samples_positive))

        events = video.get_events(False)
        tst_samples_positive, _ = create_dataset_sequentially(
            video.signal, params.sr, video.events, 
            from_time=video.val_from_time, 
            till_time=video.val_till_time, 
            window_length=params.window_length
        )

        X_tst.extend(tst_samples_positive)
        y_tst.extend([1] * len(tst_samples_positive))
    
    n_trn_negative_per_file = round(len(X_trn) / len(trn_files))
    n_tst_negative_per_file = round(len(X_tst) / len(trn_files))

    # break

    for file in trn_files:
        video = datapool[file]
        events = video.get_events(True)
        
        trn_samples_negative, labels = create_dataset_sequentially(
            video.signal, params.sr, video.events, 
            from_time=video.trn_from_time, 
            till_time=video.trn_till_time, 
            window_length=params.window_length
        )

        trn_samples_negative = np.random.choice(trn_samples_negative, n_trn_negative_per_file)
        X_trn.extend(trn_samples_negative)
        y_trn.extend([0] * len(trn_samples_negative))

        n_trn_negative = len(trn_samples_negative)

        events = video.get_events(False)
        tst_samples_negative, _ = create_dataset_sequentially(
            video.signal, params.sr, video.events, 
            from_time=video.val_from_time, 
            till_time=video.val_till_time, 
            window_length=params.window_length
        )
        n_tst_negative = len(tst_samples_negative)

        # tst_samples_negative = np.random.choice(tst_samples_negative, n_tst_negative_per_file)
        X_tst.extend(tst_samples_negative)
        y_tst.extend([0] * len(tst_samples_negative))
        # break

    # print(len(X_tst), len(y_tst), len(X_trn), len(y_trn))
    # break

    X_trn = np.array([
        transformation(x).flatten().numpy() for x in X_trn
    ])

    X_tst = np.array([
        transformation(x).flatten().numpy() for x in X_tst
    ])

    svm = LinearSVC(random_state=0, tol=1e-2, verbose=1)
    svm.fit(X_trn, y_trn)

    trn_acc = (y_trn == svm.predict(X_trn)).mean()
    tst_acc = (y_tst == svm.predict(X_tst)).mean()

    print(tst_files[0])
    print('trn_acc', trn_acc)
    print('tst_acc', tst_acc)
    print()

    table.append([tst_files[0], f'{trn_acc:.2f}', f'{tst_acc:.2f}', len(X_trn), len(X_tst)])

    # break
# %%

print(tabulate(table, headers=['file', 'trn acc', 'tst acc', 'n_trn (1:1)', 'n_tst (1:24)'],
    tablefmt='fancy_grid', showindex=True))
# %%
