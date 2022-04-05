from dataclasses import dataclass

import numpy as np
from numpy.random import normal


@dataclass
class Variant:
    lamda: float
    gamma: float
    beta_self: float
    alpha: float
    frequency: float
    parent: int = None


def create_register(l, g, B, a, f):
    """Given parameters, it creates a pokedex of variants"""
    variants = []
    for items in zip(l, g, np.diagonal(B), a, f):
        items = [round(x.item(), 5) for x in items]
        variants.append(Variant(*items, None))
    return variants


def augment_parameters(l, g, B, a, f, timer, parent_idx):
    """Adding new parameters derived from the parent index"""
    l = np.concatenate([l, l[parent_idx] + normal(size=(1, 1)) / 10]).clip(min=0)
    g = np.concatenate([g, g[parent_idx] + normal(size=(1, 1)) / 10]).clip(min=0)
    a = np.concatenate([a, a[parent_idx] + normal(size=(1, 1)) / 10]).clip(min=0)
    f = np.concatenate([f, f[parent_idx] + normal(size=(1, 1)) / 10]).clip(min=1e-6)
    timer = np.concatenate([timer, np.random.exponential(scale=1 / f[-1], size=(1, 1))])

    size = B.shape[0]
    new_B = np.zeros((size + 1, size + 1))
    new_B[:size, :size] = B
    B = new_B
    B[-1, :] = B[-2, :] + normal(size=(size + 1,)) / 10
    B[:, -1] = B[:, -2] + normal(size=(size + 1,)) / 10
    B[-1, -1] = 0
    B = B.clip(min=0)
    return l, g, B, a, f, timer


def delete_parameters(l, g, B, a, f, timer, idxes):
    """Deleting parameters at given indexes"""
    l = np.delete(l, idxes, axis=0)
    g = np.delete(g, idxes, axis=0)
    a = np.delete(a, idxes, axis=0)
    f = np.delete(f, idxes, axis=0)
    timer = np.delete(timer, idxes, axis=0)
    B = np.delete(B, idxes, axis=0)
    B = np.delete(B, idxes, axis=1)
    return l, g, B, a, f, timer
