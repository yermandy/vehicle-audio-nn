from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from model.multi import *
from utils import *


def run(files, frame_length=2.0, n_trn_samples=1000):
    uuid=int(datetime.now().timestamp())

    TRN_FROM_TIME = 1 * 60
    TRN_TILL_TIME = 25 * 60
    VAL_FROM_TIME = 25 * 60
    VAL_TILL_TIME = 34 * 60

    cuda = 0
    n_epochs = 350
    batch_size = 64
    lr = 0.0001
    n_trn_samples = n_trn_samples
    n_val_samples = 1000
    num_workers = 0

    # define parameters
    params = EasyDict()
    # hop length between nn inputs in features
    params.nn_hop_length = frame_length
    # length of one frame in seconds
    params.frame_length = frame_length
    # number of frames in one window
    params.n_frames = 1
    # length of one feature in samples
    params.n_fft = 1024
    # number of mel features
    params.n_mels = 64
    # number of mfcc features
    params.n_mfcc = 8
    # sampling rate
    params.sr = 44100
    # hop length between samples for feature extractor
    params.hop_length = 128
    # length of one window in seconds
    params.window_length = params.nn_hop_length * (params.n_frames - 1) + params.frame_length

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')

    print(f'Running on {device}')

    trn_dataset = VehicleDataset(
        files,
        from_time=TRN_FROM_TIME,
        till_time=TRN_TILL_TIME,
        params=params,
        n_samples=n_trn_samples
    )

    # trn_dataset = SyntheticDataset(
    #     signal, events, from_time=TRN_FROM_TIME, till_time=TRN_TILL_TIME, params=params, n_samples=n_trn_samples
    # )

    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = VehicleDataset(
        files,
        from_time=VAL_FROM_TIME,
        till_time=VAL_TILL_TIME,
        params=params,
        n_samples=n_val_samples
    )

    # val_dataset = SyntheticDataset(
    #     signal, events, from_time=VAL_FROM_TIME, till_time=VAL_TILL_TIME, params=params, n_samples=n_val_samples
    # )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    model = ResNet18(num_classes=10).to(device)

    loss = nn.CrossEntropyLoss()

    # optim = AdamW(model.parameters(), lr=lr)
    optim = Adam(model.parameters(), lr=lr)

    val_loss_best = float('inf')
    val_mae_best = float('inf')
    val_diff_best = float('inf')

    config = wandb.config
    config.update(params)
    config.uuid = uuid
    config.batch_size = batch_size
    config.lr = lr
    config.model = model.__class__.__name__
    config.optim = optim.__class__.__name__
    config.uniform_sampling = False if n_trn_samples == -1 else True
    config.n_trn_samples = len(trn_dataset)
    config.n_val_samples = len(val_dataset)

    wandb.run.name = str(uuid)

    audio_file = f'data/audio/{files[0]}.MP4.wav'
    labels_file = f'data/labels/{files[0]}.MP4.txt'

    signal = load_audio(audio_file)
    events = load_events(labels_file)

    params.trn = get_additional_params(
        params, signal, events, start_time=TRN_FROM_TIME, end_time=TRN_TILL_TIME
    )
    
    params.val = get_additional_params(
        params, signal, events, start_time=VAL_FROM_TIME, end_time=VAL_TILL_TIME
    )

    training_loop = tqdm(range(n_epochs))
    for _ in training_loop:

        # training
        trn_loss = 0
        trn_mae = 0

        model.train()
        for tensor, target in trn_loader:

            tensor = tensor.to(device)
            target = target.to(device)
            
            scores = model(tensor)
            loss_value = loss(scores, target)

            preds = scores.argmax(1)

            trn_loss += loss_value.detach().item()
            trn_mae += (target - preds).abs().sum().item()

            optim.zero_grad()
            loss_value.backward()
            optim.step()

        trn_mae /= len(trn_dataset)
        
        # validation
        val_loss = 0
        val_mae = 0

        model.eval()
        with torch.no_grad():
            for tensor, target in val_loader:
                tensor = tensor.to(device)
                target = target.to(device)

                scores = model(tensor)
                loss_value = loss(scores, target)

                preds = scores.argmax(1)

                val_loss += loss_value.detach().item()
                val_mae += (target - preds).abs().sum().item()

        val_mae /= len(val_dataset)

        trn_results = validate_multi(model, trn_dataset, params.trn)
        trn_diff = get_diff(trn_results, params.trn)

        val_results = validate_multi(model, val_dataset, params.val)
        val_diff = get_diff(val_results, params.val)

        if val_diff < val_diff_best:
            val_diff_best = val_diff
            torch.save(model.state_dict(), f'weights/multi/model_{uuid}_diff.pth')

        if val_loss < val_loss_best:
            val_loss_best = val_loss

        if val_mae < val_mae_best:
            val_mae_best = val_mae
            torch.save(model.state_dict(), f'weights/multi/model_{uuid}_mae.pth')


        wandb.log({
            "trn loss": trn_loss,
            "val loss": val_loss,
            "val loss best": val_loss_best,
            
            "trn mae": trn_mae,
            "val mae": val_mae,
            "val mae best": val_mae_best,

            "trn diff": trn_diff,
            "val diff": val_diff,            
            "val diff best": val_diff_best
        })

        training_loop.set_description(f'trn loss {trn_loss:.2f} | val loss {val_loss:.2f} | best loss {val_loss_best:.2f}')

        if trn_loss <= 1e-8 or trn_mae <= 1e-8:
            break


if __name__ == "__main__":

    os.makedirs('weights/multi', exist_ok=True)

    files = [
        '20190819-Kutna Hora-L1-out-MVI_0007',
        '20190819-Kutna Hora-L2-in-MVI_0030',
        '20190819-Kutna Hora-L3-in-MVI_0005',
        '20190819-Kutna Hora-L4-in-MVI_0013',
        '20190819-Kutna Hora-L5-in-MVI_0003',
        '20190819-Kutna Hora-L6-out-MVI_0017',
        '20190819-Kutna Hora-L7-out-MVI_0032',
        '20190819-Kutna Hora-L8-in-MVI_0045',
        '20190819-Kutna Hora-L9-in-MVI_0043',
        '20190819-Kutna Hora-L10-in-MVI_0029',
        '20190819-Kutna Hora-L10-out-SDV_1888',
        '20190819-Kutna Hora-L16-out-MVI_0003',
        '20190819-Kutna Hora-L18-in-MVI_0030',
        '20190819-Ricany-L2-in-MVI_0006',
        '20190819-Ricany-L2-out-MVI_0005',
        '20190819-Ricany-L6-out-MVI_0011',
        '20190819-Ricany-L7-out-MVI_0013',
        '20190819-Ricany-L8-in-MVI_0009',
        '20190819-Ricany-L9-in-MVI_0008'
    ]

    # files = ['20190819-Kutna Hora-L4-out-MVI_0040']
    # files = ['20190819-Kutna Hora-L3-in-MVI_0005']
    

    # audio_file = f'data/audio/{file}.MP4.wav'
    # labels_file = f'data/labels/{file}.MP4.txt'

    for n_trn_samples in [250, 500, 1000, 2000, 4000]:
        
        wandb_run = wandb.init(project='vehicle-audio-nn', entity='yermandy', tags=['multi'])


        wandb.config.files = files

        run(files, frame_length=6, n_trn_samples=n_trn_samples)

        wandb_run.finish()