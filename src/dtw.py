import numpy as np


def discretize(time, events):
    closest = lambda array, value: np.abs(array - value).argmin()
    return np.array([closest(time, e) for e in events])


def pairs(array):
    for i in range(len(array)):
        if i < len(array) - 1:
            yield array[i : i + 2]


def dist(t_plus_delta_0, t_plus_delta_1, cumsum, c, a=1):
    N = len(cumsum)

    if t_plus_delta_0 < 0 or t_plus_delta_0 > N:
        return np.inf

    if t_plus_delta_1 < 0 or t_plus_delta_1 > N:
        return np.inf

    if t_plus_delta_1 - t_plus_delta_0 <= 0:
        return np.inf

    # return np.abs(cumsum[t_plus_delta_0: t_plus_delta_1] - c).sum()
    f = np.concatenate(([cumsum[0]], cumsum[1:] - cumsum[:-1]))

    return (
        1 * np.abs(cumsum[t_plus_delta_0:t_plus_delta_1] - c).sum()
        + a * f[t_plus_delta_0 + 1 : t_plus_delta_1].sum()
    )


def objective(discrete_events, cumsum):
    obj = dist(0, discrete_events[0], cumsum, 0)

    for c, (i, j) in enumerate(pairs(discrete_events)):
        d = dist(i, j, cumsum, c + 1)
        obj += d

    last = len(discrete_events)
    obj += dist(discrete_events[-1], len(cumsum), cumsum, last)

    return obj


def backtrack(start, backtracking):
    backtracked = [start]
    for i in reversed(range(backtracking.shape[1] - 1)):
        start = backtracking[start, i + 1]
        backtracked.append(start)
    return backtracked[::-1]


def dtw(discrete_events, cumsum, N=5):
    deltas = np.arange(-N, N + 1)

    distances = np.zeros((len(deltas), len(discrete_events)))
    backtracking = np.zeros_like(distances, dtype=int)

    t_1 = discrete_events[0]
    t_n = discrete_events[-1]
    end = len(cumsum)
    last = len(discrete_events)

    for i, delta_1 in enumerate(deltas):
        t_plus_delta_1 = t_1 + delta_1
        d = dist(0, t_plus_delta_1, cumsum, 0)
        distances[i, 0] = d

    backtracking[:, 0] = range(len(backtracking))

    for c, (t_0, t_1) in enumerate(pairs(discrete_events)):
        for j, delta_1 in enumerate(deltas):
            min_d = np.inf

            for i, delta_0 in enumerate(deltas):
                t_plus_delta_0 = t_0 + delta_0
                t_plus_delta_1 = t_1 + delta_1

                d = dist(t_plus_delta_0, t_plus_delta_1, cumsum, c + 1)

                prev_d = distances[i, c]
                next_d = d + prev_d

                if next_d < min_d:
                    min_d = next_d
                    backtracking[j, c + 1] = i
                    distances[j, c + 1] = min_d

    for i, delta_0 in enumerate(deltas):
        t_plus_delta_0 = t_n + delta_0
        prev_d = distances[i, -1]
        d = dist(t_plus_delta_0, end, cumsum, last)
        distances[i, -1] = prev_d + d

    start = np.argmin(distances[:, -1])
    backtracked = backtrack(start, backtracking)

    new_discrete_events = []

    for delta_index, discrete_event in zip(backtracked, discrete_events):
        new_discrete_event = discrete_event + deltas[delta_index]
        new_discrete_events.append(new_discrete_event)

    new_cumstep = np.zeros_like(cumsum)
    for idx in new_discrete_events:
        new_cumstep[idx] += 1

    new_cumstep = np.cumsum(new_cumstep)
    new_discrete_events = np.array(new_discrete_events)

    return new_cumstep, new_discrete_events
