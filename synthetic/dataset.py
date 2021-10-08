import numpy as np
from collections import defaultdict
from itertools import product

def create_dataset(signal, sr, manual_events, from_time, till_time, window_length=10, n_samples=100, n_events_max=5):
    np.random.seed(42)

    samples_from = np.random.rand(1000) * (till_time - from_time - window_length) + from_time
    samples_till = samples_from + window_length

    # create intervals
    event_intervals = np.stack([samples_from, samples_till], axis=1)

    # discard interval out of time range
    # NOT NEEDED WHEN: till_time - from_time - window_length
    # mask = samples_till < till_time
    # event_intervals = event_intervals[mask];

    # add margin to discard interavls with events near borders
    margin = 1

    n_events_in_interval = []
    valid_intervals = []
    
    for event_interval in event_intervals:
        event_from, event_to = event_interval

        events_timestep = []
        events_in_interval = 0
        for manual_event in manual_events:
            if event_from <= manual_event <= event_to:
                events_in_interval += 1
                events_timestep.append(manual_event)    

        # skip interval with event in margin
        skip = False
        for event_timestep in events_timestep:
            if event_timestep < event_from + margin or event_timestep > event_to - margin:
                skip = True
                break
        if skip:
            continue

        n_events_in_interval.append(events_in_interval)
        valid_intervals.append(event_interval)

    n_events_in_interval = np.array(n_events_in_interval)
    valid_intervals = np.array(valid_intervals)

    unique_events, unique_counts = np.unique(n_events_in_interval, return_counts=True)
    
    prior_distribution = unique_counts / np.sum(unique_counts)
    prior_distribution = {k: v for k, v in zip(unique_events, prior_distribution)}
    # print(prior_distribution)

    # plt.bar(*np.unique(n_events_in_interval, return_counts=True));
    
    # matrix of pairs
    n_events_in_pairs = n_events_in_interval + n_events_in_interval[:, np.newaxis]

    pairs = []
    distribution = []
    for i, j in product(range(len(n_events_in_pairs)), range(len(n_events_in_pairs))):
        if i > j:
            continue

        n_events_in_pair = n_events_in_pairs[i, j]
        
        # number of events for diagonal pairs (same) is half of all
        n_events_in_pair = n_events_in_pair // 2 if i == j else n_events_in_pair
    
        distribution.append(n_events_in_pair)
        pairs.append((i, j))

    distribution = np.array(distribution)
    pairs = np.array(pairs)

    indices = np.arange(len(pairs))
    np.random.seed(42)
    np.random.shuffle(indices)

    distribution = distribution[indices]
    pairs = pairs[indices]
    
    # plt.bar(*np.unique(distribution, return_counts=True));
    
    n_events_intervals = defaultdict(lambda: [])
    dataset = defaultdict(lambda: [])

    for n_events, pair in zip(distribution, pairs):

        if n_events > n_events_max:
            continue

        if n_events in prior_distribution:
            
            # add to the dataset while keeping the distribution the same as the prior one
            if len(n_events_intervals[n_events]) < n_samples * prior_distribution[n_events]:
                n_events_intervals[n_events].append(valid_intervals[pair])

                interval_1, interval_2 = pair
                interval_1 = (valid_intervals[interval_1] * sr).astype(int)
                interval_2 = (valid_intervals[interval_2] * sr).astype(int)

                s1 = signal[interval_1[0]: interval_1[1]]
                s2 = signal[interval_2[0]: interval_2[1]]
                s3 = (s1 + s2) / 2
                dataset[n_events].append(s3)
        
    dataset = {k: dataset[k] for k in sorted(dataset)}
    
    return dataset

# dataset = create_dataset(signal, sr, from_time, till_time, n_samples=1000)