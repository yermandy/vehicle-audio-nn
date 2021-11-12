from torch.optim import Adam
from torch.utils.data import DataLoader

from model.classification import *
from src import *


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
        

def run(files, n_trn_samples=-1, n_val_samples=-1):
    uuid=int(datetime.now().timestamp())

    split_ratio = 0.75
    cuda = 0
    n_epochs = 1000
    batch_size = 128
    lr = 0.0001
    n_trn_samples = n_trn_samples
    n_val_samples = n_val_samples
    num_workers = 0
    use_offset = False
    normalization = Normalization.ROW_WISE
    num_classes = 50

    params = get_params(normalization=normalization)

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')

    print(f'Running on {device}')

    datapool = DataPool(files, params.window_length, split_ratio)

    trn_dataset = VehicleDataset(
        datapool,
        is_trn=True,
        params=params,
        n_samples=n_trn_samples
    )

    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = VehicleDataset(
        datapool,
        is_trn=False,
        params=params,
        n_samples=n_val_samples
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    model = ResNet18(num_classes=num_classes).to(device)

    loss = nn.CrossEntropyLoss()

    optim = Adam(model.parameters(), lr=lr)

    config = wandb.config
    config.update(params)
    config.uuid = uuid
    config.num_classes = num_classes
    config.batch_size = batch_size
    config.lr = lr
    config.model = model.__class__.__name__
    config.optim = optim.__class__.__name__
    config.uniform_sampling = False if n_trn_samples == -1 else True
    config.n_trn_samples = len(trn_dataset)
    config.n_val_samples = len(val_dataset)
    config.use_offset = use_offset

    wandb.run.name = str(uuid)

    val_loss_best = float('inf')
    val_mae_best = float('inf')
    val_diff_best = float('inf')
    val_interval_error_best = float('inf')


    training_loop = tqdm(range(n_epochs))
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

        # trn_interval_error, trn_diff = validate_intervals(datapool, True, model, trn_dataset.transform, params)
        val_interval_error, val_diff = validate_intervals(datapool, False, model, val_dataset.transform, params)

        if val_loss <= val_loss_best:
            val_loss_best = val_loss

        if val_diff <= val_diff_best:
            val_diff_best = val_diff
            torch.save(model.state_dict(), f'weights/classification/model_{uuid}_diff.pth')

        if val_mae <= val_mae_best:
            val_mae_best = val_mae
            torch.save(model.state_dict(), f'weights/classification/model_{uuid}_mae.pth')

        if val_interval_error <= val_interval_error_best:
            val_interval_error_best = val_interval_error
            torch.save(model.state_dict(), f'weights/classification/model_{uuid}_interval.pth')

        torch.save(model.state_dict(), f'weights/classification/model_{uuid}_last.pth')

        wandb.log({
            "trn loss": trn_loss,
            "val loss": val_loss,
            "val loss best": val_loss_best,

            "trn mae": trn_mae,
            "val mae": val_mae,
            "val mae best": val_mae_best,

            # "trn interval error": trn_interval_error,
            "val interval error": val_interval_error,
            "val interval error best": val_interval_error_best,

            # "trn diff": trn_diff,
            "val diff": val_diff,            
            "val diff best": val_diff_best
        })

        training_loop.set_description(f'trn loss {trn_loss:.2f} | val loss {val_loss:.2f} | best loss {val_loss_best:.2f}')

        if trn_loss <= 1e-8 or trn_mae <= 1e-8:
            break

        if use_offset:
            offset = (0.25 * iteration) % params.window_length
            trn_dataset.set_offset(offset)


if __name__ == "__main__":

    os.makedirs('weights/classification', exist_ok=True)

    files = [
        '20190819-Kutna Hora-L1-out-MVI_0007',
        '20190819-Kutna Hora-L3-in-MVI_0005',
        '20190819-Kutna Hora-L3-out-MVI_0008',
        '20190819-Kutna Hora-L4-in-MVI_0013',
        '20190819-Kutna Hora-L6-out-MVI_0017',
        '20190819-Kutna Hora-L7-out-MVI_0032',
        '20190819-Kutna Hora-L8-in-MVI_0045',
        '20190819-Kutna Hora-L9-in-MVI_0043',
        '20190819-Kutna Hora-L10-out-SDV_1888',
        '20190819-Kutna Hora-L13-in-MVI_0006',
        '20190819-Kutna Hora-L13-out-MVI_0018',
        '20190819-Kutna Hora-L14-out-MVI_0005',
        '20190819-Kutna Hora-L15-out-MVI_0012',
        '20190819-Kutna Hora-L18-in-MVI_0030',
        '20190819-Ricany-L2-in-MVI_0006',
        '20190819-Ricany-L2-out-MVI_0005',
        '20190819-Ricany-L3-in-MVI_0006',
        '20190819-Ricany-L6-in-MVI_0008',
        '20190819-Ricany-L6-out-MVI_0011',
        '20190819-Ricany-L7-in-MVI_0008',
        '20190819-Ricany-L7-out-MVI_0013',
        '20190819-Ricany-L8-in-MVI_0009',
        '20190819-Ricany-L8-out-MVI_0013',
        '20190819-Ricany-L9-in-MVI_0008',
        '20190819-Ricany-L9-out-MVI_0011'
    ]

    files = ['20190819-Kutna Hora-L4-out-MVI_0040_manual']

    for n_trn_samples in [-1]:
        
        wandb_run = wandb.init(project='vehicle-audio-nn', entity='yermandy', tags=['classification'])

        wandb.config.files = files

        run(files, n_trn_samples=n_trn_samples, n_val_samples=-1)

        wandb_run.finish()