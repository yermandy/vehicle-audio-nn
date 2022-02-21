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


def forward(loader, model, loss, optim=None, is_train=False):
    device = next(model.parameters()).device
    n_samples = len(loader.dataset)

    loss_sum = 0
    abs_error_sum = 0

    n_events_true = defaultdict(lambda: 0)
    n_events_pred = defaultdict(lambda: 0)

    if is_train:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(is_train):
        for tensor, target, domain in loader:
            tensor = tensor.to(device)
            target = target.to(device)
            logits = model(tensor)

            loss_value = loss(logits, target)
            loss_sum += loss_value.item()
            
            preds = logits.argmax(1)
            abs_error_sum += (target - preds).abs().sum().item()

            if optim is not None:
                optim.zero_grad()
                loss_value.backward()
                optim.step()
            
            for d, t, p in zip(domain.tolist(), target.tolist(), preds.tolist()):
                n_events_true[d] += t
                n_events_pred[d] += p

    rvce = np.mean([abs(n_events_true[d] - n_events_pred[d]) / n_events_true[d] for d in n_events_true])
    mae = abs_error_sum / n_samples
    
    return mae, loss_sum, rvce


def validate_and_save(uuid, datapool, prefix='tst', is_trn=None, model_name='rvce'):
    model, config = load_model_locally(uuid, model_name)
    
    outputs = validate_datapool(datapool, model, config, is_trn)
    table, fancy_table = create_fancy_table(outputs)
    with open(f'outputs/{uuid}/results/{prefix}_{model_name}_output.txt', 'w') as file:
        file.write(fancy_table)
    np.savetxt(f'outputs/{uuid}/results/{prefix}_{model_name}_output.csv', table, fmt='%s', delimiter=';')


@hydra.main(config_path='config', config_name='default')
def run(config: DictConfig):
    print_config(config)

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

    trn_datapool = DataPool(config.training_files, config.window_length, config.split_ratio)

    trn_dataset = VehicleDataset(
        trn_datapool,
        is_trn=True,
        config=config,
        n_samples=config.n_trn_samples
    )

    trn_loader = DataLoader(trn_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    val_dataset = VehicleDataset(
        trn_datapool,
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
    val_rvce_best = float('inf')

    with open(f'outputs/{uuid}/config.pickle', 'wb') as f:
        pickle.dump(config, f)

    training_loop = tqdm(range(config.n_epochs))
    for iteration in training_loop:

        ## training
        trn_mae, trn_loss, trn_rvce = forward(trn_loader, model, loss, optim, True)

        ## validation
        # trn_mae, trn_loss, trn_rvce = forward(trn_loader, model, loss)
        val_mae, val_loss, val_rvce = forward(val_loader, model, loss)

        ## calculate rvce from sequentioal data
        # trn_rvce = validate_intervals(trn_datapool, True, model, trn_dataset.transform, config)
        # val_rvce = validate_intervals(trn_datapool, False, model, val_dataset.transform, config)

        if val_loss <= val_loss_best:
            val_loss_best = val_loss

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

            "trn rvce": trn_rvce,
            "val rvce": val_rvce,
            "val rvce best": val_rvce_best,
        })

        training_loop.set_description(f'trn loss {trn_loss:.2f} | val loss {val_loss:.2f} | best loss {val_loss_best:.2f}')

        if trn_loss <= 1e-8 or trn_mae <= 1e-8:
            print('finishing earlier')
            break

        if config.use_offset:
            offset = (config.offset_length * iteration) % config.window_length
            trn_dataset.set_offset(offset)

    os.makedirs(f'outputs/{uuid}/results/', exist_ok=True)
    
    validate_and_save(uuid, trn_datapool, 'val', False, 'rvce')
    validate_and_save(uuid, trn_datapool, 'val', False, 'mae')
    
    validate_and_save(uuid, trn_datapool, 'trn', True, 'rvce')
    validate_and_save(uuid, trn_datapool, 'trn', True, 'mae')

    if len(config.testing_files) > 0:
        tst_datapool = DataPool(config.testing_files, config.window_length, config.split_ratio)
        validate_and_save(uuid, tst_datapool, 'tst', None, 'rvce')
        validate_and_save(uuid, tst_datapool, 'tst', None, 'mae')

    wandb_run.finish()


def setup_hydra():
    sys.argv.append(r'hydra.run.dir=outputs/${uuid}')
    sys.argv.append(f'hydra.output_subdir=config')
    sys.argv.append(f'hydra/job_logging=disabled')
    sys.argv.append(f'hydra/hydra_logging=none')

if __name__ == "__main__":
    setup_hydra()
    run()