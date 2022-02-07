from train_classification import *
import yaml
import argparse


def cross_validation_error(uuids, model_name='rvce'):
    root_uuid = uuids[0].split('/')[0]
    table = []
    for uuid in uuids:
        results = np.genfromtxt(f'outputs/{uuid}/results/tst_{model_name}_output.csv', delimiter=';', skip_header=1, skip_footer=1, dtype=str)
        results = np.atleast_2d(results)
        table.extend(results)
    table = np.array(table)
    table, fancy_table = create_fancy_table(table)
    np.savetxt(f'outputs/{root_uuid}/tst_{model_name}_output.csv', table, fmt='%s', delimiter=';')
    with open(f'outputs/{root_uuid}/tst_{model_name}_output.txt', 'w') as file:
        file.write(fancy_table)


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

    with open(f'config/{args.config_name}.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
        cross_validate(config)
    
