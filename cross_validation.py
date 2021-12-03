from train_classification import *
import yaml
import argparse


def cross_validation_error(uuids):
    test_outputs = []
    for uuid in uuids:
        results = np.loadtxt(f'outputs/{uuid}/results/test_output.csv', delimiter='; ', skiprows=1, usecols=[0,1,2,3])
        results = np.atleast_2d(results)
        test_outputs.append(results.mean(0))
    outputs_std = np.mean(test_outputs, axis=0)
    outputs_mean = np.std(test_outputs, axis=0)

    header = 'rvce; error; n_events; mae'
    row = np.array([f'{i:.3f} ± {j:.3f}' for i, j in zip(outputs_mean, outputs_std)])[np.newaxis, :]

    uuid = uuid.split('/')[0]
    np.savetxt(f'outputs/{uuid}/outputs.csv', row, fmt='%s', delimiter='; ', header=header)


def cross_validate(config):
    config = EasyDict(config)

    sys.argv.append(f'training_files')
    sys.argv.append(f'testing_files')
    sys.argv.append(f'uuid')

    cross_validation_uuid = int(datetime.now().timestamp())

    uuids = []

    for split, (training_files, testing_files) in enumerate(zip(config.training_splits, config.testing_splits)):

        if 'output_name' in config:
            uuid = f'{config.output_name}/{split}'
        else:
            uuid = f'{cross_validation_uuid}/{int(datetime.now().timestamp())}'
        
        for i, arg in enumerate(sys.argv):
            if 'training_files' in arg:
                sys.argv[i] = f'+training_files={training_files}' 
            if 'testing_files' in arg:
                sys.argv[i] = f'+testing_files={testing_files}' 
            if 'uuid' in arg:
                sys.argv[i] = f'+uuid={uuid}' 

        uuids.append(uuid)
        sys.argv.append(f'hydra.run.dir=outputs/{uuid}')

        run()

    cross_validation_error(uuids)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default='config_cross_validation', type=str)
    args = parser.parse_args()

    sys.argv.extend([f'--config-name', f'{args.config_name}'])
    sys.argv.append(f'hydra.output_subdir=config')
    sys.argv.append(f'hydra/job_logging=disabled')
    sys.argv.append(f'hydra/hydra_logging=none')

    with open('config/config_cross_validation.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
        cross_validate(config)
    