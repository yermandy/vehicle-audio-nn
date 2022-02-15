from .utils import *
from .transformation import *


def validate(signal, model, transform, config, tqdm=lambda x: x, return_probs=False, from_time=None, till_time=None, classification=True):

    if from_time is not None and till_time is not None:
        signal = crop_signal(signal, config.sr, from_time, till_time)
        
    device = next(model.parameters()).device

    batch = []
    predictions = []
    probs = []

    n_hops = get_n_hops(signal, config) 

    loop = tqdm(range(n_hops))

    model.eval()
    with torch.no_grad():
        for k in loop:
            start = k * config.n_samples_in_nn_hop
            end = start + config.n_samples_in_window
            x = signal[start: end]
            x = transform(x)
            batch.append(x)
            
            if (k + 1) % config.batch_size == 0 or k + 1 == n_hops:
                batch = torch.stack(batch, dim=0)
                batch = batch.to(device)
                scores = model(batch)
                
                if return_probs:
                    p = scores.softmax(1).tolist()
                    probs.extend(p)

                if classification:
                    y = scores.argmax(1).view(-1).tolist()
                else:
                    y = scores.view(-1).tolist()

                predictions.extend(y)
                batch = []

    predictions = np.array(predictions)

    if return_probs:
        probs = np.array(probs)
        return predictions, probs

    return predictions


def validate_intervals(datapool: DataPool, is_trn: bool, model, transform, params, classification=True):
    rvce = 0
    n_intervals = 0

    for video in datapool:
        video: Video = video
        n_events = video.get_events_count(is_trn)
        from_time, till_time = video.get_from_till_time(is_trn)

        predictions = validate(video.signal, model, transform, params, from_time=from_time, till_time=till_time, classification=classification)
        n_intervals += 1

        rvce += np.abs(predictions.sum() - n_events) / n_events

    mean_rvce = rvce / n_intervals
    return mean_rvce


def validate_datapool(datapool, model, config, is_trn=None):
    """
        Returns array [[0:rvce, 1:error, 2:n_events, 3:mae, 4:time, 5:file]]
    """
    transform = create_transformation(config)
    outputs = []

    for video in tqdm(datapool):
        if is_trn is None:
            from_time = 0
            till_time = len(video.signal) // config.sr
            n_events = len(video.events)
        else:
            n_events = video.get_events_count(is_trn)
            from_time, till_time = video.get_from_till_time(is_trn)

        predictions = validate(video.signal, model, transform, config, from_time=from_time, till_time=till_time, classification=True)
        labels = get_labels(video.events, config.window_length, from_time, till_time)
        mae = np.abs(predictions - labels).mean()

        n_predicted = predictions.sum()
        rvce = np.abs(n_predicted - n_events) / n_events
        error = n_predicted - n_events
        outputs.append([f'{rvce:.3f}', error, n_events, f'{mae:.3f}', f'[{from_time:.0f}, {till_time:.0f}]', video.file])

    outputs = np.array(outputs)
    idx = np.argsort(outputs[:, 0].astype(float))
    outputs = outputs[idx]

    # rvce = outputs[:, 0].astype(float).mean()
    # print('rvce', rvce)
    
    return outputs
