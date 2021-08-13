from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from model.regression import *
from utils import *

torch.backends.cudnn.benchmark = True


def run(audio_file, labels_file, nn_hop_length=5.0, frame_length=2.0, n_frames=3):
    uuid=int(datetime.now().timestamp())

    TRN_FROM_TIME = 1 * 60
    TRN_TILL_TIME = 25 * 60
    VAL_FROM_TIME = 25 * 60
    VAL_TILL_TIME = 34 * 60

    cuda = 1
    n_epochs = 500
    batch_size = 64
    lr = 0.0001

    # define parameters
    params = EasyDict()
    # hop length between nn inputs in features
    params.nn_hop_length = nn_hop_length
    # length of one frame in seconds
    params.frame_length = frame_length
    # number of frames in one window
    params.n_frames = n_frames
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

    signal = load_audio(audio_file)
    events = load_events(labels_file)

    trn_dataset = VehicleDataset(
        signal,
        events,
        start_time=TRN_FROM_TIME,
        end_time=TRN_TILL_TIME,
        use_offset=True,
        params=params
    )

    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = VehicleDataset(
        signal,
        events,
        start_time=VAL_FROM_TIME,
        end_time=VAL_TILL_TIME,
        seed=0,
        params=params
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = ResNet18().to(device)
    # model = MobileNetV3S().to(device)
    # model = MobileNetV3L().to(device)

    optim = AdamW(model.parameters(), lr=lr)

    val_loss_best = float('inf')
    val_smallest_diff = float('inf')

    config = wandb.config
    config.update(params)
    config.uuid = uuid
    config.batch_size = batch_size
    config.lr = lr
    config.model = model.__class__.__name__

    wandb.run.name = str(uuid)

    split_indices = get_split_indices(params)

    params.trn = get_additional_params(
        params, signal, events, start_time=TRN_FROM_TIME, end_time=TRN_TILL_TIME
    )
    
    params.val = get_additional_params(
        params, signal, events, start_time=VAL_FROM_TIME, end_time=VAL_TILL_TIME
    )

    # val_dataset_2 = VehicleValidationDataset(params.val.signal, params.val, val_dataset.transform)
    # val_loader_2 = DataLoader(val_dataset_2, batch_size=batch_size)

    training_loop = tqdm(range(n_epochs))
    for _ in training_loop:

        # training
        trn_loss = 0
        n_processed = 0

        model.train()
        for tensor, target in trn_loader:

            tensor = tensor.to(device)
            target = target.to(device)

            n_samples = tensor.shape[0]
            n_processed += n_samples

            Y = torch.zeros(n_samples).to(device)

            for start, end in split_indices:
                x = tensor[..., start: end]
                Y += model(x).squeeze()

            loss_value = (Y - target).abs().sum()

            trn_loss += loss_value.detach().item()

            optim.zero_grad()
            loss_value.backward()
            optim.step()

        trn_loss /= len(trn_dataset) * len(split_indices)
        # creates a new offset
        trn_loader.dataset.split_signal()

        # validation
        val_loss = 0
        n_processed = 0

        model.eval()
        with torch.no_grad():
            for tensor, target in val_loader:
                tensor = tensor.to(device)
                target = target.to(device)

                n_samples = tensor.shape[0]
                n_processed += n_samples * len(split_indices)

                Y = torch.zeros(n_samples).to(device)

                for start, end in split_indices:
                    x = tensor[..., start: end]
                    Y += model(x).squeeze()

                loss_value = (Y - target).abs().sum()

                val_loss += loss_value.detach().item()

        val_loss /= len(trn_dataset) * len(split_indices)

        # trn_results = validate(model, trn_dataset, params.trn)
        # trn_results = validate(model, trn_dataset, params.trn)
        # trn_diff = get_diff(trn_results, params.trn)

        val_results = validate(model, val_dataset, params.val)
        val_diff = get_diff(val_results, params.val)

        if val_diff < val_smallest_diff:
            val_smallest_diff = val_diff
            torch.save(model.state_dict(), f'weights/regression/model_{uuid}_diff.pth')

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            torch.save(model.state_dict(), f'weights/regression/model_{uuid}_loss.pth')

        wandb.log({
            "trn loss": trn_loss,
            "val loss": val_loss,
            # "trn diff": trn_diff,
            "val diff": val_diff,
            "val loss best": val_loss_best,
            "val smallest diff": val_smallest_diff
        })

        training_loop.set_description(
            f'trn loss {trn_loss:.2f} | val loss {val_loss:.2f} | best loss {val_loss_best:.2f}')


if __name__ == "__main__":

    os.makedirs('weights/regression', exist_ok=True)

    audio_file = 'data/audio/20190819-Kutna Hora-L4-out-MVI_0040.wav'
    labels_file = 'data/labels/20190819-Kutna Hora-L4-out-MVI_0040.txt'

    # '''
    for nn_hop_length in [0.75, 1]:
        for frame_length in [1, 1.5]:
            for n_frames in [16]:
                if nn_hop_length > frame_length:
                    continue

                print('nn_hop_length:', nn_hop_length)
                print('frame_length:', frame_length)
                print('n_frames:', n_frames)

                wandb_run = wandb.init(project='vehicle-audio-nn', entity='yermandy')
                # wandb_run = wandb.init(project='vehicle-audio-nn', entity='yermandy', tags=['grid search'])

                run(audio_file, labels_file, nn_hop_length=nn_hop_length,
                    frame_length=frame_length, n_frames=n_frames)

                wandb_run.finish()
    # '''

    '''
    # wandb_run = wandb.init(project='test', entity='yermandy', tags=['test'])
    for i in range(3):
        wandb_run = wandb.init(project='vehicle-audio-nn', entity='yermandy')
        run(audio_file, labels_file, nn_hop_length=3, frame_length=3, n_frames=3)
        wandb_run.finish()
    # '''
