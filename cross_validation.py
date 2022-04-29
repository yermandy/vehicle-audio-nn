from train_classification import *


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
    
