from typing import Callable, Dict, Tuple
from .utils import *
from .transformation import *
from scipy.special import logsumexp


def validate_video(video: Video, model, return_probs=True, return_preds=True,
                   from_time=None, till_time=None, classification=True):

    signal = video.signal
    config = video.config
    transform = create_transformation(config)

    if from_time is None:
        from_time = 0

    if till_time is None:
        till_time = get_signal_length(signal, config)

    if from_time is not None and till_time is not None:
        signal = crop_signal(signal, config.sr, from_time, till_time)
        
    device = next(model.parameters()).device

    batch = []
    preds = defaultdict(list)
    probs = defaultdict(list)

    n_hops = get_n_hops(config, from_time, till_time) 

    model.eval()
    with torch.no_grad():
        for k in range(n_hops):
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

                    if return_preds:
                        if classification:
                            y = scores.argmax(1).flatten().tolist()
                        else:
                            y = scores.flatten().tolist()
                        preds[head].extend(y)
                
                batch = []

    to_return = []

    if return_preds:
        preds = {k: np.array(v) for k, v in preds.items()}
        to_return.append(preds)
    
    if return_probs:
        probs = {k: np.array(v) for k, v in probs.items()}
        to_return.append(probs)
    
    return to_return if len(to_return) > 1 else to_return[0]


def validate_datapool(datapool: DataPool, model, config: Config, part=Part.WHOLE):
    dict = defaultdict(list)
    files = []
    times = []

    for video in tqdm(datapool):
        from_time, till_time = video.get_from_till_time(part)

        files.append(video.file)
        times.append(f'{from_time:.0f}: {till_time:.0f}')

        probs = validate_video(video, model, return_preds=False, from_time=from_time, till_time=till_time, classification=True)
        preds, n_predicted = inference(probs, config)
        labels = get_labels(video, from_time, till_time)

        for head in config.heads:
            head_labels = labels[head]
            head_n_events = head_labels.sum()

            if preds is not None:
                head_preds = preds[head]
                head_mae = np.abs(head_preds - head_labels).mean()
                dict[f'mae: {head}'].append(f'{head_mae:.4f}')
            
            if n_predicted is not None:
                head_n_predicted = n_predicted[head]
                # TODO think about it
                if head_n_events == 0:
                    head_rvce = np.abs(head_n_predicted - head_n_events)
                else:
                    head_rvce = np.abs(head_n_predicted - head_n_events) / head_n_events
                head_error = head_n_predicted - head_n_events
                dict[f'rvce: {head}'].append(f'{head_rvce:.4f}')
                dict[f'n_events: {head}'].append(head_n_events)
                dict[f'error: {head}'].append(f'{head_error}')

    append_summary(dict, times, files)
    
    return dict


def append_summary(dict, times, files):
    for k, v in dict.items():
        v = np.array(v).astype(float)
        stats = f'{v.mean():.2f} ± {v.std():.2f}'
        dict[k].append(stats)

    times.append('')
    files.append('summary')

    dict['time'] = times
    dict['file'] = files


def validate_and_save(uuid, datapool, prefix='tst', part=Part.WHOLE, model_name='rvce', config=None):
    model, _config = load_model_locally(uuid, model_name)
    if config is None:
        config = _config
    datapool_summary = validate_datapool(datapool, model, config, part)
    save_dict_txt(f'outputs/{uuid}/results/{prefix}_{model_name}_output.txt', datapool_summary)
    save_dict_csv(f'outputs/{uuid}/results/{prefix}_{model_name}_output.csv', datapool_summary)


def inference_simple(probs: Dict[str, np.ndarray]):
    preds = {}
    n_predicted = {}
    for head, head_probs in probs.items():
        preds[head] = head_probs.argmax(1)
        n_predicted[head] = preds[head].sum()
    return preds, n_predicted


def total_count_distribution(p_count):
    """ 
    Input:
      p_c [n_labels x n_windows] 
    Output:
      distr [n_windows * (n_labels - 1)] distr[c] is the probability 
        that the total number of events is c 
    """

    n_labels, n_windows = p_count.shape

    log_p_count = np.log(p_count)
    log_P = np.zeros(((n_labels - 1) * n_windows + 1, n_windows))
    log_P.fill(np.NINF)

    for s in range(n_labels):
        log_P[s, n_windows - 1] = np.log(p_count[s, n_windows - 1])
        
    for i in range(n_windows - 2, -1, -1):
        for s in range((n_windows - i) * (n_labels - 1) + 1):
            a = []
            for c in range(max(s - (n_windows - i - 1) * (n_labels - 1), 0), min(n_labels, s + 1)):
                a.append(log_p_count[c, i] + log_P[s - c, i + 1])
            log_P[s, i] = logsumexp(a)

    return np.exp(log_P[:, 0]) 


def optimal_rvce_predictor(probs, dist):
    n_labels, n_windows = probs.shape
    max_total_count = (n_labels - 1) * n_windows
    rvce_risk = np.zeros(max_total_count)
    count_range = np.arange(1, max_total_count + 1)
    for c in range(1, max_total_count + 1):
        rvce_risk[c - 1] = np.sum(dist[1:] * np.abs(count_range - c) / count_range)
    pred_count_rvce = np.argmin(rvce_risk) + 1
    return pred_count_rvce


def inference_optimal_rvce(probs: Dict[str, np.ndarray]):
    print('optimal')
    n_predicted = {}
    for head, head_probs in probs.items():
        head_probs = head_probs.T
        dist = total_count_distribution(head_probs)
        n_predicted[head] = optimal_rvce_predictor(head_probs, dist)
    return None, n_predicted


def inference(preds: Dict[str, np.ndarray], config: Config) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    if config.inference_function.is_simple():
        return inference_simple(preds)
    elif config.inference_function.is_optimal_rvce():
        return inference_optimal_rvce(preds)