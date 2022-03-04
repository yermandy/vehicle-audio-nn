from train_classification import *


def generate_cross_validation_table(uuids, model_name='rvce', prefix='tst'):
    table = []
    for uuid in uuids:
        results = np.genfromtxt(f'outputs/{uuid}/results/{prefix}_{model_name}_output.csv', delimiter=';', skip_header=1, skip_footer=1, dtype=str)
        results = np.atleast_2d(results)
        table.extend(results)
    table = np.array(table)
    table, fancy_table = create_fancy_table(table)
    np.savetxt(f'outputs/{root_uuid}/{prefix}_{model_name}_output.csv', table, fmt='%s', delimiter=';')
    with open(f'outputs/{root_uuid}/{prefix}_{model_name}_output.txt', 'w') as file:
        file.write(fancy_table)


@hydra.main(config_path='config', config_name='default')
def setup_globals(config: DictConfig):
    global cross_validation_folder, root_uuid, n_splits
    cross_validation_folder = config['cross_validation_folder']
    root_uuid = config['uuid']
    n_splits = config['n_splits']


def setup_hydra():
    sys.argv.append(f'hydra.output_subdir=config')
    sys.argv.append(f'hydra/job_logging=disabled')
    sys.argv.append(f'hydra/hydra_logging=none')
    sys.argv.append('hydra.run.dir=outputs/tmp')


def cross_validate():
    setup_globals()

    uuids = []
    for split in range(n_splits):
        split_uuid = f'{root_uuid}/{split}'
        uuids.append(split_uuid)
        sys.argv.append(f'++split={split}')
        sys.argv.append(f'uuid={split_uuid}')
        sys.argv.append(f'hydra.run.dir=outputs/{split_uuid}')
        sys.argv.append(f'training_files={cross_validation_folder}/{split}')
        sys.argv.append(f'testing_files={cross_validation_folder}/{split}')
        run()
    
    generate_cross_validation_table(uuids, 'rvce')
    generate_cross_validation_table(uuids, 'mae')

if __name__ == "__main__":
    setup_hydra()
    cross_validate()
    
