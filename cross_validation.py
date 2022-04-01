from train_classification import *


def generate_cross_validation_table(uuids, model_name='rvce', prefix='tst'):
    root_uuid = uuids[0].split('/')[0]
    table = []
    header = []
    dict = {}
    for uuid in uuids:
        results = np.genfromtxt(f'outputs/{uuid}/results/{prefix}_{model_name}_output.csv', delimiter=',', skip_footer=1, dtype=str)
        header = results[0]
        results = results[1:]
        results = np.atleast_2d(results)
        table.extend(results)
    table = np.array(table).T
    times = []
    files = []
    for i in range(len(header)):
        column_name = header[i]
        column = table[i].tolist()
        if column_name == 'file':
            files = column
        elif column_name == 'time':
            times = column
        else:
            dict[column_name] = column

    append_summary(dict, times, files)
    save_dict_csv(f'outputs/{root_uuid}/{prefix}_{model_name}_output.csv', dict)
    save_dict_txt(f'outputs/{root_uuid}/{prefix}_{model_name}_output.txt', dict)


@hydra.main(config_path='config', config_name='default')
def setup_globals(_config):
    global config
    config = Config(_config)


def setup_hydra():
    sys.argv.append(f'hydra.output_subdir=config')
    sys.argv.append(f'hydra/job_logging=disabled')
    sys.argv.append(f'hydra/hydra_logging=none')
    sys.argv.append('hydra.run.dir=outputs/tmp')


def cross_validate():
    setup_hydra()
    setup_globals()

    global config

    uuids = []
    for split in range(config.splits_from, config.n_splits):
        print(split)
        split_uuid = f'{config.uuid}/{split}'
        uuids.append(split_uuid)
        sys.argv.append(f'++split={split}')
        sys.argv.append(f'++root_uuid={config.uuid}')
        sys.argv.append(f'uuid={split_uuid}')
        sys.argv.append(f'hydra.run.dir=outputs/{split_uuid}')
        sys.argv.append(f'training_files={config.cross_validation_folder}/{split}')
        sys.argv.append(f'testing_files={config.cross_validation_folder}/{split}')
        sys.argv.append(f'validation_files={config.cross_validation_folder}/{split}')
        run()
    
    generate_cross_validation_table(uuids, 'rvce')
    generate_cross_validation_table(uuids, 'mae')

if __name__ == "__main__":
    cross_validate()
    
