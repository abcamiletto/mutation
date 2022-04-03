import numpy as np


def model(t, X, l, g, a, B):
    """Calculates dX/dt and returns it"""
    S, I, R, W = unpack(X)

    dSdt = -l.T @ I * S
    dIdt = l * I * S - g * I + B.T @ W * I
    dRdt = g * I - a * R
    dWdt = a * R - B @ I * W

    dXdt = [dSdt, dIdt, dRdt, dWdt]
    return pack(dXdt)


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
