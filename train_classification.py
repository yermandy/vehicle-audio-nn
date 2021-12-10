from torch.optim import Adam
from torch.utils.data import DataLoader

from model.classification import *
from src import *
from omegaconf import DictConfig, OmegaConf

import hydra
import sys
import pickle
import yaml
import argparse


def forward(loader, model, loss):
    device = next(model.parameters()).device
    n_samples = len(loader.dataset)

    loss_sum = 0
    abs_error_sum = 0

    model.eval()
    with torch.no_grad():
        for tensor, target in loader:
            tensor = tensor.to(device)
            target = target.to(device)

            scores = model(tensor)
            loss_value = loss(scores, target)

            preds = scores.argmax(1)

            loss_sum += loss_value.detach().item()
            abs_error_sum += (target - preds).abs().sum().item()
    
    mae = abs_error_sum / n_samples
    return mae, loss_sum


def save_csv(model, datapool, prefix='test', model_name='rvce'):
    if prefix == 'trn':
        is_trn = True
    elif prefix == 'val':
        is_trn = False
    else:
        is_trn = None
    
    model, run_config = load_model_locally(uuid, model_name)
    outputs = validate_datapool(datapool, model, run_config, is_trn)
    table = print_validation_outputs(outputs)
    np.savetxt(f'outputs/{uuid}/results/{prefix}_output.txt', table, fmt='%s')
    header = 'rvce; error; n_events; mae; file'
    np.savetxt(f'outputs/{uuid}/results/{prefix}_output.csv', outputs, fmt='%s', delimiter='; ', header=header)


@hydra.main(config_path='config', config_name='config')
def run(config: DictConfig):
    wandb_run = wandb.init(project=config.wandb_project, entity=config.wandb_entity, tags=config.wandb_tags)

    # replace DictConfig with EasyDict
    config = OmegaConf.to_container(config)
    config = EasyDict(config)

    # get uuid and change wandb run name
    uuid = config.uuid
    wandb.run.name = str(uuid)
    os.makedirs(f'weights')

    # set original root
    root = hydra.utils.get_original_cwd()
    os.chdir(root)

    config = get_additional_params(config)

    device = torch.device(f'cuda:{config.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')

    datapool = DataPool(config.training_files, config.window_length, config.split_ratio)

    trn_dataset = VehicleDataset(
        datapool,
        is_trn=True,
        config=config,
        n_samples=config.n_trn_samples
    )

    trn_loader = DataLoader(trn_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    val_dataset = VehicleDataset(
        datapool,
        is_trn=False,
        config=config,
        n_samples=config.n_val_samples
    )

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers)

    model = ResNet18(num_classes=config.num_classes).to(device)

    loss = nn.CrossEntropyLoss()

    optim = Adam(model.parameters(), lr=config.lr)

    wandb_config = wandb.config
    wandb_config.update(config)
    wandb_config.uuid = uuid
    wandb_config.model = model.__class__.__name__
    wandb_config.optim = optim.__class__.__name__
    wandb_config.uniform_sampling = False if config.n_trn_samples == -1 else True
    wandb_config.update({'n_trn_samples': len(trn_dataset)}, allow_val_change=True)
    wandb_config.update({'n_val_samples': len(val_dataset)}, allow_val_change=True)


    val_loss_best = float('inf')
    val_mae_best = float('inf')
    val_diff_best = float('inf')
    val_rvce_best = float('inf')

    with open(f'outputs/{uuid}/config.pickle', 'wb') as f:
        pickle.dump(config, f)

    training_loop = tqdm(range(config.n_epochs))
    for iteration in training_loop:

        ## training
        trn_loss = 0
        # trn_mae = 0

        model.train()
        for tensor, target in trn_loader:

            tensor = tensor.to(device)
            target = target.to(device)
            
            scores = model(tensor)
            loss_value = loss(scores, target)

            trn_loss += loss_value.detach().item()
            # preds = scores.argmax(1)
            # trn_mae += (target - preds).abs().sum().item()

            optim.zero_grad()
            loss_value.backward()
            optim.step()
        # trn_mae = trn_mae / len(trn_dataset)

        ## validation
        trn_mae, trn_loss = forward(trn_loader, model, loss)
        val_mae, val_loss = forward(val_loader, model, loss)

        # trn_interval_error, trn_diff = validate_intervals(datapool, True, model, trn_dataset.transform, config)
        val_rvce, val_diff = validate_intervals(datapool, False, model, val_dataset.transform, config)

        if val_loss <= val_loss_best:
            val_loss_best = val_loss

        if val_diff <= val_diff_best:
            val_diff_best = val_diff
            torch.save(model.state_dict(), f'outputs/{uuid}/weights/chd.pth')

        if val_mae <= val_mae_best:
            val_mae_best = val_mae
            torch.save(model.state_dict(), f'outputs/{uuid}/weights/mae.pth')

        if val_rvce <= val_rvce_best:
            val_rvce_best = val_rvce
            torch.save(model.state_dict(), f'outputs/{uuid}/weights/rvce.pth')

        torch.save(model.state_dict(), f'outputs/{uuid}/weights/last.pth')

        wandb.log({
            "trn loss": trn_loss,
            "val loss": val_loss,
            "val loss best": val_loss_best,

            "trn mae": trn_mae,
            "val mae": val_mae,
            "val mae best": val_mae_best,

            # "trn rvce": trn_rvce,
            "val rvce": val_rvce,
            "val rvce best": val_rvce_best,

            # "trn diff": trn_diff,
            "val diff": val_diff,            
            "val diff best": val_diff_best
        })

        training_loop.set_description(f'trn loss {trn_loss:.2f} | val loss {val_loss:.2f} | best loss {val_loss_best:.2f}')

        if trn_loss <= 1e-8 or trn_mae <= 1e-8:
            print('finishing earlier')
            break

        if config.use_offset:
            offset = (config.offset_length * iteration) % config.window_length
            trn_dataset.set_offset(offset)

    os.makedirs(f'outputs/{uuid}/results/', exist_ok=True)
    
    save_csv(model, datapool, 'val')
    save_csv(model, datapool, 'trn')

    if len(config.testing_files) > 0:
        datapool = DataPool(config.testing_files, config.window_length, config.split_ratio)
        save_csv(model, datapool, 'tst')

    wandb_run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default='config', type=str)
    args = parser.parse_args()

    sys.argv.append(f'hydra.output_subdir=config')
    sys.argv.append(f'hydra/job_logging=disabled')
    sys.argv.append(f'hydra/hydra_logging=none')

    with open(f'config/{args.config_name}.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
        config = EasyDict(config)

    if 'output_name' in config and config.output_name != '' and config.output_name is not None:
        uuid = config.output_name
    else:
        uuid = int(datetime.now().timestamp())
    print('Run name:', uuid)
    
    sys.argv.append(f'+uuid={uuid}')
    sys.argv.append(f'hydra.run.dir=outputs/{uuid}')
    # sys.argv.append(f'training_files=dataset_26.11.2021')
    # sys.argv.append(f'training_files=manual')
    
    run()