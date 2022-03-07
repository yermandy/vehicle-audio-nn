from .utils import *
from .transformation import *


def validate_video(video: Video, model, tqdm=lambda x: x, return_probs=False, from_time=None, till_time=None, classification=True):

    signal = video.signal
    config = video.config
    transform = create_transformation(config)

    if from_time is not None and till_time is not None:
        signal = crop_signal(signal, config.sr, from_time, till_time)
        
    device = next(model.parameters()).device

    batch = []
    preds = defaultdict(list)
    probs = defaultdict(list)

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
                heads = model(batch)
                
                for head, scores in heads.items():
                    if return_probs:
                        p = scores.softmax(1).tolist()
                        probs[head].extend(p)

                    if classification:
                        y = scores.argmax(1).flatten().tolist()
                    else:
                        y = scores.flatten().tolist()

                    preds[head].extend(y)
                
                batch = []

    preds = {k: np.array(v) for k, v in preds.items()}

    if return_probs:
        probs = {k: np.array(v) for k, v in probs.items()}
        return preds, probs

    return preds


def validate_datapool(datapool: DataPool, model, config, part=Part.TEST):
    """
        Returns array [[0:rvce, 1:error, 2:n_events, 3:mae, 4:time, 5:file]]
    """
    outputs = []

    for video in tqdm(datapool):
        n_events = video.get_events_count(part)
        from_time, till_time = video.get_from_till_time(part)

        preds = validate_video(video, model, from_time=from_time, till_time=till_time, classification=True)
        preds = preds['n_counts']

        labels = get_labels(video.events, config.window_length, from_time, till_time)
        mae = np.abs(preds - labels).mean()

        n_predicted = preds.sum()
        rvce = np.abs(n_predicted - n_events) / n_events
        error = n_predicted - n_events
        outputs.append([f'{rvce:.3f}', error, n_events, f'{mae:.3f}', f'[{from_time:.0f}, {till_time:.0f}]', video.file])

    outputs = np.array(outputs)
    idx = np.argsort(outputs[:, 0].astype(float))
    outputs = outputs[idx]

    # rvce = outputs[:, 0].astype(float).mean()
    # print('rvce', rvce)
    
    return outputs
