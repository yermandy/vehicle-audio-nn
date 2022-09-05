from typing import Callable, Dict, Tuple
from .utils import *
from .transformation import *
from scipy.special import logsumexp
from .seqevents import Events as SeqEvents
from .seqevents_general import Events as SeqEventsGeneral


def validate_video(
    video: Video,
    model: torch.nn.Module,
    return_probs=True,
    return_preds=True,
    from_time=None,
    till_time=None,
    classification=True,
    tqdm=lambda x: x,
):

    signal = video.signal
    config = video.config
    transform = create_transformation(config)

    if from_time is None:
        from_time = 0

    max_signal_length = get_signal_length(signal, config)
    if till_time is None or till_time > max_signal_length:
        print(f"Till time is {till_time} but max signal length is {max_signal_length}")
        till_time = max_signal_length

    signal = crop_signal(signal, config.sr, from_time, till_time)

    device = next(model.parameters()).device

    batch = []
    preds = defaultdict(list)
    probs = defaultdict(list)

    n_hops = get_n_hops(config, from_time, till_time)

    model.eval()
    with torch.no_grad():
        for k in tqdm(range(n_hops)):
            start = k * config.n_samples_in_nn_hop
            end = start + config.n_samples_in_window
            x = signal[start:end]
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


def change_probs_for_doubled_inference(probs_1, probs_2):
    n_events = 17

    Px1 = probs_1["n_counts"][:, :n_events].T
    Px2 = probs_2["n_counts"][:, :n_events].T

    Pc = np.empty((Px1.shape[0], Px1.shape[1] + Px2.shape[1]))
    Pc[:, 0::2] = Px1
    Pc[:, 1::2] = Px2

    Pc = Pc / Pc.sum(0)

    return {"n_counts": Pc}


def get_probs_for_dense_inference(
    video: Video, model, from_time, till_time, n_windows_for_dense_inference: int
):
    Pxs = []

    offset = video.config.window_length / n_windows_for_dense_inference

    height = 0
    width = 0

    for i in range(n_windows_for_dense_inference):
        probs = validate_video(
            video,
            model,
            return_preds=False,
            from_time=from_time + offset,
            till_time=till_time,
        )
        probs = probs["n_counts"]
        height += probs.shape[0]
        width = probs.shape[1]
        Pxs.append(probs)

    Pc = np.empty((height, width))

    for i, Px in enumerate(Pxs):
        Pc[i::n_windows_for_dense_inference] = Px

    return {"n_counts": Pc}


def collect_probs_from_models(video: Video, models, from_time, till_time):
    probs_ensembled = defaultdict(int)
    for model in models:
        if video.config.inference_function.is_doubled():
            probs = get_probs_for_dense_inference(video, model, from_time, till_time, 2)
        elif video.config.inference_function.is_dense():
            probs = get_probs_for_dense_inference(
                video,
                model,
                from_time,
                till_time,
                video.config.n_windows_for_dense_inference,
            )
        else:
            probs = validate_video(
                video,
                model,
                return_preds=False,
                from_time=from_time,
                till_time=till_time,
            )
        for head, head_probs in probs.items():
            probs_ensembled[head] += head_probs / head_probs.sum(1, keepdims=True)
    # make a valid probablility distribution
    for head in video.config.heads:
        probs_ensembled[head] = probs_ensembled[head] / probs_ensembled[head].sum(
            1, keepdims=True
        )
    return probs_ensembled


def validate_datapool(datapool: DataPool, model, config: Config, part=Part.WHOLE):
    dict = defaultdict(list)
    files = []
    times = []

    for video in tqdm(datapool):
        video: Video = video
        from_time, till_time = video.get_from_till_time(part)

        files.append(video.file)
        times.append(f"{from_time:.0f}: {till_time:.0f}")

        # in case of ensembling, we get the predictions for each model
        if type(model) == list:
            print("Ensembling")
            probs = collect_probs_from_models(video, model, from_time, till_time)
        elif config.inference_function.is_doubled():
            probs = get_probs_for_dense_inference(video, model, from_time, till_time, 2)
        elif config.inference_function.is_dense():
            probs = get_probs_for_dense_inference(
                video, model, from_time, till_time, config.n_windows_for_dense_inference
            )
        else:
            probs = validate_video(
                video,
                model,
                return_preds=False,
                from_time=from_time,
                till_time=till_time,
            )

        preds, n_predicted = inference(probs, config)
        labels = get_labels(video, from_time, till_time)

        for head in config.heads:
            head_labels = labels[head]

            if config.use_manual_counts and head == "n_counts":
                head_n_events = video.manual_counts
            else:
                head_n_events = head_labels.sum()

            if preds is not None:
                head_preds = preds[head]
                head_mae = np.abs(head_preds - head_labels).mean()
                dict[f"mae: {head}"].append(f"{head_mae:.4f}")

            if n_predicted is not None:
                head_n_predicted = n_predicted[head]
                # TODO think about it
                if head_n_events == 0:
                    head_rvce = np.abs(head_n_predicted - head_n_events)
                else:
                    head_rvce = np.abs(head_n_predicted - head_n_events) / head_n_events
                head_error = head_n_predicted - head_n_events
                dict[f"rvce: {head}"].append(f"{head_rvce:.4f}")
                dict[f"n_events: {head}"].append(head_n_events)
                dict[f"error: {head}"].append(f"{head_error}")

    append_summary(dict, times, files)

    return dict


def validate_and_save(
    uuid, datapool, prefix="tst", part=Part.WHOLE, model_name="rvce", config=None
):
    model, _config = load_model_locally(uuid, model_name)
    if config is None:
        config = _config
    datapool_summary = validate_datapool(datapool, model, config, part)
    save_dict_txt(
        f"outputs/{uuid}/results/{prefix}_{model_name}_output.txt", datapool_summary
    )
    save_dict_csv(
        f"outputs/{uuid}/results/{prefix}_{model_name}_output.csv", datapool_summary
    )


# by xfrancv
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
            for c in range(
                max(s - (n_windows - i - 1) * (n_labels - 1), 0), min(n_labels, s + 1)
            ):
                a.append(log_p_count[c, i] + log_P[s - c, i + 1])
            log_P[s, i] = logsumexp(a)

    return np.exp(log_P[:, 0])


# by xfrancv
def struct_inference(log_Pc, log_P):
    """
    Input:
        log_Pc [n_classes x n_windows] log probability of the total number of events
        log_P list of tuples([n_classes x n_windows]), i.e. log_P[0][0].shape = (n_classes, n_windows)
    Output:
        c - labels for each window of counting head
        lab - labels for each window of coupled heads
    """
    n_events = log_Pc.shape[0] - 1
    n_wins = log_Pc.shape[1]

    phi = []
    arg_phi = []
    score = np.copy(log_Pc)
    for i in range(len(log_P)):
        log_Px = log_P[i][0]
        log_flip_Py = np.flipud(log_P[i][1])

        phi_ = np.zeros((n_events + 1, n_wins))
        arg_phi_ = np.zeros((n_events + 1, n_wins), dtype=int)
        for c in range(n_events + 1):
            tmp = log_Px[0 : c + 1, :] + log_flip_Py[-(c + 1) :, :]
            arg_phi_[c, :] = np.argmax(tmp, axis=0)
            # phi_[c,:] = np.max( tmp, axis=0)
            idx_row, idx_col = np.unravel_index(
                arg_phi_[c, :] * tmp.shape[1] + np.arange(0, tmp.shape[1]), tmp.shape
            )
            phi_[c, :] = tmp[idx_row, idx_col]

        arg_phi.append(arg_phi_)

        score += phi_

    c = np.argmax(score, axis=0)

    lab = []
    for i in range(len(log_P)):
        idx_row, idx_col = np.unravel_index(
            c * n_wins + np.arange(0, n_wins), (n_events + 1, n_wins)
        )
        lab.append(arg_phi[i][idx_row, idx_col])

    return c, lab


def optimal_rvce_predictor(probs, dist):
    n_labels, n_windows = probs.shape
    max_total_count = (n_labels - 1) * n_windows
    rvce_risk = np.zeros(max_total_count)
    count_range = np.arange(1, max_total_count + 1)
    for c in range(1, max_total_count + 1):
        rvce_risk[c - 1] = np.sum(dist[1:] * np.abs(count_range - c) / count_range)
    pred_count_rvce = np.argmin(rvce_risk) + 1
    return pred_count_rvce


def inference_simple(probs: Dict[str, np.ndarray]):
    preds = {}
    n_predicted = {}
    for head, head_probs in probs.items():
        preds[head] = head_probs.argmax(1)
        n_predicted[head] = preds[head].sum()
    return preds, n_predicted


def inference_optimal_rvce(probs: Dict[str, np.ndarray]):
    print("optimal")
    n_predicted = {}
    for head, head_probs in probs.items():
        head_probs = head_probs.T
        dist = total_count_distribution(head_probs)
        n_predicted[head] = optimal_rvce_predictor(head_probs, dist)
    return None, n_predicted


def inference_doubled(probs: Dict[str, np.ndarray]):
    print("doubled")
    n_predicted = {}
    for head, head_probs in probs.items():
        n_events_max = 17
        head_probs = head_probs[:, :n_events_max].T
        head_probs = head_probs / head_probs.sum(0)
        n_events, seq_len = head_probs.shape
        A = SeqEvents(n_events // 2, seq_len + 1)
        est_Px, est_Pc, kl_hist = A.deconv(head_probs)
        n_predicted[head] = est_Px.argmax(0).sum()
    # TODO reconstruct predictions
    return None, n_predicted


def inference_dense(probs: Dict[str, np.ndarray], config: Config):
    print("dense")
    n_predicted = {}
    for head, head_probs in probs.items():
        n_events = int(
            config.n_windows_for_dense_inference * config.n_events_per_dense_window + 1
        )
        head_probs = head_probs[:, :n_events].T
        head_probs = head_probs / head_probs.sum(0)
        A = SeqEventsGeneral(config.n_windows_for_dense_inference)
        est_Px, est_Pc, kl_hist = A.deconv(head_probs, 30)
        n_predicted[head] = est_Px.argmax(0).sum()
    return None, n_predicted


def inference_structured(probs: Dict[str, np.ndarray], config: Config):
    print("structured")
    preds = {}
    n_predicted = {}

    log_Pc = np.log(probs["n_counts"]).T

    log_P = []
    for label_1, label_2 in config.coupled_labels:
        log_P.append([np.log(probs[label_1]).T, np.log(probs[label_2]).T])

    c, lab = struct_inference(log_Pc, log_P)

    preds["n_counts"] = c
    n_predicted["n_counts"] = c.sum()

    for i, (label_1, label_2) in enumerate(config.coupled_labels):
        preds[label_1] = lab[i]
        preds[label_2] = c - lab[i]

        n_predicted[label_1] = preds[label_1].sum()
        n_predicted[label_2] = preds[label_2].sum()

    return preds, n_predicted


def inference(
    preds: Dict[str, np.ndarray], config: Config
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    if config.inference_function.is_simple():
        return inference_simple(preds)
    elif config.inference_function.is_optimal_rvce():
        return inference_optimal_rvce(preds)
    elif config.inference_function.is_doubled():
        return inference_doubled(preds)
    elif config.inference_function.is_dense():
        return inference_dense(preds, config)
    elif config.inference_function.is_structured():
        return inference_structured(preds, config)
