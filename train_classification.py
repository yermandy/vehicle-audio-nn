from torch.optim import Adam
from torch.utils.data import DataLoader

from src import *
from omegaconf import DictConfig, OmegaConf

import hydra
import sys
import pickle
import yaml
import argparse


def forward(loader, model, loss, config, optim=None, is_train=False):
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
        for tensor, labels in loader:
            tensor = tensor.to(device)
            heads = model(tensor)
            
            domain = labels['domain']

            loss_value = 0

            for head, head_logits in heads.items():

                head_labels = labels[head].to(device)
                head_weight = config.heads[head]

                loss_value += head_weight * loss(head_logits, head_labels)
                loss_sum += loss_value.item()
                
                if head == 'n_counts':
                    head_preds = head_logits.argmax(1)
                    abs_error_sum += (head_labels - head_preds).abs().sum().item()

                    for d, t, p in zip(domain.tolist(), head_labels.tolist(), head_preds.tolist()):
                        n_events_true[d] += t
                        n_events_pred[d] += p

            if optim is not None and is_train:
                optim.zero_grad()
                loss_value.backward()
                optim.step()

    rvce = np.mean([abs(n_events_true[d] - n_events_pred[d]) / n_events_true[d] for d in n_events_true])
    mae = abs_error_sum / n_samples
    
    return mae, loss_sum, rvce


@hydra.main(config_path='config', config_name='default')
def run(config):
    # make config type and attribute safe
    config = Config(config)

    # print config
    print_config(config)

    # initialize wandb run
    wandb_run = wandb.init(project=config.wandb_project, entity=config.wandb_entity, tags=config.wandb_tags)

    # get uuid and change wandb run name
    uuid = config.uuid
    wandb.run.name = str(uuid)
    os.makedirs(f'weights')

    # set original root
    root = hydra.utils.get_original_cwd()
    os.chdir(root)

    # set device
    device = get_device(config.cuda)

    # initialize training datapool
    trn_datapool = DataPool(config.training_files, config)

    # initialize training dataset
    trn_dataset = VehicleDataset(trn_datapool, part=Part.TRAINING, config=config)
    trn_loader = DataLoader(trn_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)

    # initialize validation dataset
    val_dataset = VehicleDataset(trn_datapool, part=Part.VALIDATION, config=config)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers)

    # initialize model
    model = ResNet18(config).to(device)

    # initialize loss function
    loss = nn.CrossEntropyLoss()

    # initialize optimizer
    optim = Adam(model.parameters(), lr=config.lr)

    config.n_trn_samples = len(trn_dataset)
    config.n_val_samples = len(val_dataset)
    config.model = model.__class__.__name__
    config.optim = optim.__class__.__name__
    wandb.config.update(config)

    val_loss_best = float('inf')
    val_mae_best = float('inf')
    val_rvce_best = float('inf')

    with open(f'outputs/{uuid}/config.pickle', 'wb') as f:
        pickle.dump(config, f)

    training_loop = tqdm(range(config.n_epochs))
    for iteration in training_loop:

        ## training
        trn_mae, trn_loss, trn_rvce = forward(trn_loader, model, loss, config, optim, True)

        ## validation
        # trn_mae, trn_loss, trn_rvce = forward(trn_loader, model, loss, config,)
        val_mae, val_loss, val_rvce = forward(val_loader, model, loss, config)

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
            trn_dataset.create_with_offset(offset)

    os.makedirs(f'outputs/{uuid}/results/', exist_ok=True)
    
    validate_and_save(uuid, trn_datapool, 'val', Part.VALIDATION, 'rvce')
    validate_and_save(uuid, trn_datapool, 'val', Part.VALIDATION, 'mae')
    
    validate_and_save(uuid, trn_datapool, 'trn', Part.TRAINING, 'rvce')
    validate_and_save(uuid, trn_datapool, 'trn', Part.TRAINING, 'mae')

    if len(config.testing_files) > 0:
        tst_datapool = DataPool(config.testing_files, config)
        validate_and_save(uuid, tst_datapool, 'tst', Part.TEST, 'rvce')
        validate_and_save(uuid, tst_datapool, 'tst', Part.TEST, 'mae')

    wandb_run.finish()


def setup_hydra():
    sys.argv.append(r'hydra.run.dir=outputs/${uuid}')
    sys.argv.append(f'hydra.output_subdir=config')
    sys.argv.append(f'hydra/job_logging=disabled')
    sys.argv.append(f'hydra/hydra_logging=none')

if __name__ == "__main__":
    setup_hydra()
    run()