import numpy as np


def unpack(X):
    size = round((X.shape[0] - 1) / 3)
    S = X[0, None]  # equivalent of S = np.expand_dims(X[0], 1)
    I = X[1 : size + 1, None]
    R = X[size + 1 : size * 2 + 1, None]
    W = X[size * 2 + 1 : size * 3 + 1, None]
    return S, I, R, W


def pack(list_of_elements):
    array = np.concatenate(list_of_elements)
    return np.squeeze(array)


def delete_states(X, idxes):
    """Deletes states at given indexes"""
    size = round((X.shape[0] - 1) / 3)
    state_idx = []
    for idx in idxes:
        state_idx.extend([idx + 1, size + idx + 1, 2 * size + idx + 1])

    X = np.delete(X, state_idx)
    return X


def append_state(X, idx, unit):
    """Append a new dimension in our state variable"""
    S, I, R, W = unpack(X)

    S = np.expand_dims(S, 1)
    I[idx] -= unit
    I = np.expand_dims(np.append(I, unit), 1)
    R = np.expand_dims(np.append(R, 0), 1)
    W = np.expand_dims(np.append(W, 0), 1)

    X = pack([S, I, R, W])
    return X


def get_abs_idx(abs_state, infected, unit=0):
    """Given the history state, we recover the absolute index of that variant from its value"""
    # Finding the absolute index of the variant
    # We use ranges and not equalities because of some numerical inconsistencies
    size = round((len(abs_state) - 1) / 3)
    abs_state = abs_state[1 : size + 1]

    upper_b = infected + 1e-9
    lower_b = infected - 1e-9

    real_idx = np.where((lower_b < abs_state) & (abs_state < upper_b))

    # It may happen that the variant to delete is one that just spawned
    if (real_idx[0].size == 0) and unit:
        # In that case we look for it like this
        target = infected + unit
        upper_b = target + 1e-9
        lower_b = target - 1e-9

        real_idx = np.where((lower_b < abs_state) & (abs_state < upper_b))

    return real_idx[0][0].item()
