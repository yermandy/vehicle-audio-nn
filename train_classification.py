from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from src import *


def forward(loader, model, loss, config, optim=None, is_train=False):
    device = next(model.parameters()).device

    loss_sum = 0
    abs_errors = defaultdict(list)

    n_events_true = defaultdict(lambda: defaultdict(int))
    n_events_pred = defaultdict(lambda: defaultdict(int))

    if is_train:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(is_train):
        for tensor, labels in loader:
            domain = labels['domain']
            tensor = tensor.to(device)
            heads = model(tensor)

            loss_value = 0

            for head, head_logits in heads.items():

                head_labels = labels[head].to(device)
                head_weight = config.heads[head]

                loss_value += head_weight * loss(head_logits, head_labels)
                loss_sum += loss_value.item()
                
                head_preds = head_logits.argmax(1)
                head_abs_errors = (head_labels - head_preds).abs().tolist()
                abs_errors[head].extend(head_abs_errors)

                for d, t, p in zip(domain.tolist(), head_labels.tolist(), head_preds.tolist()):
                    n_events_true[head][d] += t
                    n_events_pred[head][d] += p

            if optim is not None and is_train:
                optim.zero_grad()
                loss_value.backward()
                optim.step()

    # weighted rvce by head weight
    rvce = []
    for head, head_weight in config.heads.items():
        head_rvce = head_weight * np.mean([
            abs(n_events_true[head][domain] - n_events_pred[head][domain]) / n_events_true[head][domain] 
            for domain in n_events_true[head] 
            if n_events_true[head][domain] != 0
        ])
        rvce.append(head_rvce)
    rvce = np.mean(rvce)

    # weighted mae by head weight
    mae = np.mean([
        config.heads[head] * np.mean(head_abs_errors) 
        for head, head_abs_errors 
        in abs_errors.items()
    ])
    
    return loss_sum, mae, rvce


def replace(i, path, model_name):
    if i < 0:
        return
    if os.path.exists(f'{path}/{model_name}_{i}.pth'):
        shutil.copy2(f'{path}/{model_name}_{i}.pth', f'{path}/{model_name}_{i + 1}.pth')
    replace(i - 1, path, model_name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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

    use_different_validation_files = len(config.validation_files) > 0

    # initialize training and validation datapool
    val_datapool = trn_datapool = DataPool(config.training_files, config)

    if use_different_validation_files:
        val_datapool = DataPool(config.validation_files, config)

    tst_datapool = DataPool(config.testing_files, config)

    trn_part = Part.WHOLE if use_different_validation_files else Part.LEFT
    val_part = Part.WHOLE if use_different_validation_files else Part.RIGHT

    set_seed(config.seed)
    g = torch.Generator()
    g.manual_seed(config.seed)

    # initialize training dataset
    trn_dataset = VehicleDataset(trn_datapool, part=trn_part, config=config, is_trn=True)
    trn_loader = DataLoader(trn_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True, drop_last=True, worker_init_fn=seed_worker, generator=g)

    # initialize validation dataset
    val_dataset = VehicleDataset(val_datapool, part=val_part, config=config)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers)

    # initialize model
    model = get_model(config).to(device)

    # initialize loss function
    if config.loss == 'ClassBalancedCrossEntropy':
        if len(config.heads) > 1:
            raise Exception('Class-Balanced Cross Entropy is not supported for multi-head training.')
        else:
            # https://arxiv.org/pdf/1901.05555.pdf
            classes, samples = np.unique(trn_dataset.labels['n_counts'], return_counts=True)
            samples_per_class = np.ones(config.num_classes)
            samples_per_class[classes] = samples
            effective_num = 1.0 - np.power(config.loss_cbce_beta, samples_per_class)
            weights = (1.0 - config.loss_cbce_beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * len(samples_per_class)
            weights = torch.from_numpy(weights).float().to(device)
            print('ClassBalancedCrossEntropy')
            loss = nn.CrossEntropyLoss(weights)
    else:
        loss = nn.CrossEntropyLoss()
            

    # initialize optimizer
    optim = get_optimizer(model, config)

    config.n_trn_samples = len(trn_dataset)
    config.n_val_samples = len(val_dataset)
    wandb.config.update(config)

    val_loss_best = float('inf')
    val_mae_best = float('inf')
    val_rvce_best = float('inf')
    tst_rvce_best = float('inf')
    tst_mae_best = float('inf')
    tst_summary = None

    with open(f'outputs/{uuid}/config.pickle', 'wb') as f:
        pickle.dump(config, f)

    shutil.make_archive(f'outputs/{uuid}/src', 'zip', 'src')

    training_loop = tqdm(range(config.n_epochs))
    for iteration in training_loop:

        ## training
        trn_loss, trn_mae, trn_rvce = forward(trn_loader, model, loss, config, optim, True)

        ## validation
        # trn_loss, trn_mae, trn_rvce = forward(trn_loader, model, loss, config)
        val_loss, val_mae, val_rvce = forward(val_loader, model, loss, config)

        # val_summary = validate_datapool(tst_datapool, model, config, val_part)
        # val_loss, val_mae, val_rvce = 0, val_summary['mae: n_counts'], val_summary['rvce: n_counts']

        ## testing
        if config.use_testing_files:
            tst_summary = validate_datapool(tst_datapool, model, config, Part.WHOLE)

        if val_loss < val_loss_best:
            val_loss_best = val_loss

        if val_mae < val_mae_best:
            val_mae_best = val_mae
            torch.save(model.state_dict(), f'outputs/{uuid}/weights/mae.pth')

        if val_rvce < val_rvce_best:
            val_rvce_best = val_rvce
            #! To save more than one model, uncomment the following line
            # replace(4, f'outputs/{uuid}/weights', 'rvce')
            # torch.save(model.state_dict(), f'outputs/{uuid}/weights/rvce_0.pth')
            torch.save(model.state_dict(), f'outputs/{uuid}/weights/rvce.pth')

        torch.save(model.state_dict(), f'outputs/{uuid}/weights/last.pth')

        log = {
            "trn loss": trn_loss,
            "val loss": val_loss,
            "val loss best": val_loss_best,

            "trn mae": trn_mae,
            "val mae": val_mae,
            "val mae best": val_mae_best,

            "trn rvce": trn_rvce,
            "val rvce": val_rvce,
            "val rvce best": val_rvce_best,
        }

        if tst_summary != None:
            tst_rvce = tst_summary['rvce: n_counts']
            tst_mae = tst_summary['mae: n_counts']

            tst_rvce = np.mean(list(map(float, tst_rvce[:-1])))
            tst_mae = np.mean(list(map(float, tst_mae[:-1])))

            if tst_rvce < tst_rvce_best:
                tst_rvce_best = tst_rvce

            if tst_mae < tst_mae_best:
                tst_mae_best = tst_mae

            log['tst rvce'] = tst_rvce
            log['tst rvce best'] = tst_rvce_best
            log['tst mae'] = tst_mae
            log['tst mae best'] = tst_mae_best

        wandb.log(log)

        training_loop.set_description(f'trn loss {trn_loss:.2f} | val loss {val_loss:.2f} | best loss {val_loss_best:.2f}')

        if trn_loss <= 1e-8 or trn_mae <= 1e-8:
            print('finishing earlier')
            break

        if config.use_offset:
            if config.use_random_offset:
                offset = np.random.rand(1)[0] * config.window_length
            else:
                offset = (config.offset_length * iteration) % config.window_length
            trn_dataset.create_with_offset(offset)

    os.makedirs(f'outputs/{uuid}/results/', exist_ok=True)
    
    validate_and_save(uuid, val_datapool, 'val', val_part, 'rvce')
    validate_and_save(uuid, val_datapool, 'val', val_part, 'mae')
    
    validate_and_save(uuid, trn_datapool, 'trn', trn_part, 'rvce')
    validate_and_save(uuid, trn_datapool, 'trn', trn_part, 'mae')

    validate_and_save(uuid, tst_datapool, 'tst', Part.WHOLE, 'rvce')
    validate_and_save(uuid, tst_datapool, 'tst', Part.WHOLE, 'mae')

    wandb_run.finish()


def setup_hydra():
    sys.argv.append(r'hydra.run.dir=outputs/${uuid}')
    sys.argv.append(f'hydra.output_subdir=config')
    sys.argv.append(f'hydra/job_logging=disabled')
    sys.argv.append(f'hydra/hydra_logging=none')

if __name__ == "__main__":
    setup_hydra()
    run()