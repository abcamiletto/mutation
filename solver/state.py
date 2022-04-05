import numpy as np


def unpack(X):
    size = round((X.shape[0] - 1) / 3)
    S = np.expand_dims(X[0], 0)
    I = np.expand_dims(X[1 : size + 1], 1)
    R = np.expand_dims(X[size + 1 : size * 2 + 1], 1)
    W = np.expand_dims(X[size * 2 + 1 : size * 3 + 1], 1)
    return S, I, R, W


def pack(list_of_elements):
    array = np.concatenate(list_of_elements)
    return np.squeeze(array)


def delete_state():
    pass


def add_state():
    pass


def get_abs_idx(abs_state, state, unit):
    pass
