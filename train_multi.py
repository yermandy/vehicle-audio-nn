from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from model.multi import *
from utils import *


def run(audio_file, labels_file, nn_hop_length=5.0, frame_length=2.0, n_frames=3):
    uuid=int(datetime.now().timestamp())

    TRN_FROM_TIME = 1 * 60
    TRN_TILL_TIME = 25 * 60
    VAL_FROM_TIME = 25 * 60
    VAL_TILL_TIME = 34 * 60

    cuda = 0
    n_epochs = 500
    batch_size = 64

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

    model = ResNet18(num_classes=10).to(device)

    loss = nn.CrossEntropyLoss()

    optim = AdamW(model.parameters(), lr=0.0001)

    val_loss_best = float('inf')
    val_error_best = float('inf')

    config = wandb.config
    config.update(params)
    config.uuid = uuid
    config.batch_size = batch_size

    wandb.run.name = str(uuid)

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
        trn_error = 0

        model.train()
        for tensor, target in trn_loader:

            tensor = tensor.to(device)
            target = target.to(device)
            
            scores = model(tensor)
            loss_value = loss(scores, target)

            preds = scores.argmax(1)

            trn_loss += loss_value.detach().item()
            trn_error += (target - preds).abs().sum().item()

            optim.zero_grad()
            loss_value.backward()
            optim.step()

        trn_loss /= len(trn_dataset)
        trn_error /= len(trn_dataset)
        
        # creates a new offset
        trn_loader.dataset.split_signal()

        # validation
        val_loss = 0
        val_error = 0

        model.eval()
        with torch.no_grad():
            for tensor, target in val_loader:
                tensor = tensor.to(device)
                target = target.to(device)

                scores = model(tensor)
                loss_value = loss(scores, target)

                preds = scores.argmax(1)

                val_loss += loss_value.detach().item()
                val_error += (target - preds).abs().sum().item()

        val_loss /= len(trn_dataset)
        val_error /= len(trn_dataset)

        # trn_results = validate(model, trn_dataset, params.trn)
        # trn_diff = get_diff(trn_results, params.trn)

        # val_results = validate(model, val_dataset, params.val)
        # val_diff = get_diff(val_results, params.val)

        # if val_diff < val_smallest_diff:
        #     val_smallest_diff = val_diff
        #     torch.save(model.state_dict(), f'weights/multi/model_{uuid}_diff.pth')

        if val_error < val_error_best:
            val_error_best = val_error
            torch.save(model.state_dict(), f'weights/multi/model_{uuid}.pth')

        wandb.log({
            "trn loss": trn_loss,
            # "trn diff": trn_diff,
            "trn error": trn_error,

            "val loss": val_loss,
            # "val diff": val_diff,
            "val error": val_error,
            "val loss best": val_loss_best,
            # "val smallest diff": val_smallest_diff
        })

        training_loop.set_description(f'trn loss {trn_loss:.2f} | val loss {val_loss:.2f} | best loss {val_loss_best:.2f}')


if __name__ == "__main__":

    os.makedirs('weights/multi', exist_ok=True)

    audio_file = 'data/audio/20190819-Kutna Hora-L4-out-MVI_0040.wav'
    labels_file = 'data/labels/20190819-Kutna Hora-L4-out-MVI_0040.txt'

    '''
    # for nn_hop_length in [1.0, 2.0, 3.0]:
    for nn_hop_length in [2.0, 3.0]:
        for frame_length in [1.0, 2.0, 3.0]:
            for n_frames in [3, 5, 7]:
                if nn_hop_length > frame_length:
                    continue

                print('nn_hop_length:', nn_hop_length)
                print('frame_length:', frame_length)
                print('n_frames:', n_frames)

                wandb_run = wandb.init(project='vehicle-audio-nn', entity='yermandy', tags=['grid search'])

                run(audio_file, labels_file, nn_hop_length=nn_hop_length,
                    frame_length=frame_length, n_frames=n_frames)

                wandb_run.finish()
    # '''

    # '''
    wandb_run = wandb.init(project='test', entity='yermandy', tags=['test'])
    run(audio_file, labels_file, nn_hop_length=3, frame_length=3, n_frames=1)
    wandb_run.finish()
    # '''
