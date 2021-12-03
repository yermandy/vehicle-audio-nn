from .utils import *


def _validate_signal(signal, model, config, tqdm=lambda x: x, batch_size=32, return_probs=False, from_time=None, till_time=None, classification=True):
    transform = create_transformation(config)

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
            end = start + config.n_samples_in_frame
            x = signal[start: end]
            x = transform(x)
            batch.append(x)
            
            if (k + 1) % batch_size == 0 or k + 1 == n_hops:
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


def _validate_intervals(datapool: DataPool, is_trn: bool, model, config, classification=True):
    transform = create_transformation(config)

    interval_error = 0
    difference_error = 0
    n_intervals = 0
    for video in datapool:
        video: Video = video
        n_events = video.get_events_count(is_trn)
        from_time, till_time = video.get_from_till_time(is_trn)

        predictions = validate(video.signal, model, transform, config, from_time=from_time, till_time=till_time, classification=classification)
        n_intervals += 1

        # calculate error at the end of interval
        interval_error += np.abs(predictions.sum() - n_events) / n_events

        # calculate cumulative histogram difference
        difference_error += get_diff(video.signal, video.events, predictions, config, from_time, till_time)

    mean_interval_error = interval_error / n_intervals
    mean_difference_error = difference_error / n_intervals

    return mean_interval_error, mean_difference_error


def validate_datapool(datapool, model, config, is_trn=None):
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
        mae = get_diff(video.signal, video.events, predictions, config, from_time, till_time)

        n_predicted = predictions.sum()
        rvce = np.abs(n_predicted - n_events) / n_events
        error = n_predicted - n_events
        outputs.append([f'{rvce:.3f}', error, n_events, f'{mae:.3f}', f'[{from_time:.0f}, {till_time:.0f}]  {video.file}'])

    outputs = np.array(outputs)
    idx = np.argsort(outputs[:, 0].astype(float))
    outputs = outputs[idx]

    rvce = outputs[:, 0].astype(float).mean()

    print('rvce', rvce)
    
    return outputs


def print_validation_outputs(outputs):
    table = []
    table_header = ' # |  rvce | error | n_events |     mae | file'
    table_separator = '– ' * 50
    table.append(table_header)
    table.append(table_separator)
    for i, row in enumerate(outputs):
        table_row = f'{i + 1:>2} | {row[0]} | {row[1]:>5} | {row[2]:>8} | {row[3]:>7} | {row[4]}'
        table.append(table_row)
    table.append(table_separator)
    mean_rvce = outputs[:, 0].astype(float).mean()
    mean_error = outputs[:, 1].astype(int).mean()
    mean_n_events = outputs[:, 2].astype(int).mean()
    mean_mae = outputs[:, 3].astype(float).mean()
    table_summary = f'   | {mean_rvce:.3f} | {mean_error:>5} | {mean_n_events:>8} | {mean_mae:7.4} | '
    table.append(table_summary)
    for x in table:
        print(x)
    return table


def validate_and_save(uuid, datapool):
    model_name = 'rvce'
    model, run_config = load_model_locally(uuid, model_name)
    outputs = validate_datapool(datapool, model, run_config)
    table = print_validation_outputs(outputs)
    np.savetxt(f'outputs/{uuid}/test_output.txt', table, fmt='%s')
    header = 'rvce; error; n_events; mae; file'
    np.savetxt(f'outputs/{uuid}/test_output.csv', outputs, fmt='%s', delimiter=';', header=header)