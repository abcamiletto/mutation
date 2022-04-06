from dataclasses import dataclass

import numpy as np
from numpy.random import normal

from .state import pack, unpack


@dataclass
class Variant:
    lamda: float
    gamma: float
    beta_self: float
    alpha: float
    frequency: float
    parent: int = None
    I0: float = None


def create_register(l, g, B, a, f, X0):
    """Given parameters, it creates a pokedex of variants"""
    _, I0, _, _ = unpack(X0)
    variants = []
    for items in zip(l, g, np.diagonal(B), a, f, I0):
        items = [round(x.item(), 5) for x in items]
        variants.append(Variant(*items[:-1], None, items[-1]))
    return variants


def augment_beta(B, l, parent_idx=None, beta_self=None):
    """Expanding the beta matrix, given the previous beta and lambda"""
    size = B.shape[0]
    new_B = np.zeros((size + 1, size + 1))
    new_B[:size, :size] = B
    B = new_B
    B[-1, :] = np.random.uniform(size=(size + 1,)) * 0.7 * np.squeeze(l)
    B[:, -1] = np.random.uniform(size=(size + 1,)) * 0.7 * l[size]
    if parent_idx:
        B[parent_idx, -1] = B[parent_idx, -1] / 2
        B[-1, parent_idx] = B[-1, parent_idx] / 2
    B[-1, -1] = beta_self or np.random.uniform(size=(1,)) / 7
    return B.clip(min=0)


def augment_parameters(l, g, B, a, f, timer, parent_idx):
    """Adding new parameters derived from the parent index"""
    l = np.concatenate([l, l[parent_idx] + normal(size=(1, 1)) / 10]).clip(min=0)
    g = np.concatenate([g, g[parent_idx] + normal(size=(1, 1)) / 10]).clip(min=0)
    a = np.concatenate([a, a[parent_idx] + normal(size=(1, 1)) / 10]).clip(min=0)
    f = np.concatenate([f, f[parent_idx] + normal(size=(1, 1)) / 10]).clip(min=1e-6)
    timer = np.concatenate([timer, np.random.exponential(scale=1 / f[-1], size=(1, 1))])

    B = augment_beta(B, l, parent_idx=parent_idx)
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


def get_last_variant(l, g, B, a, f, timer, real_idx, I0):
    return Variant(
        round(l[-1].item(), 5),
        round(g[-1].item(), 5),
        round(B[-1, -1].item(), 5),
        round(a[-1].item(), 5),
        round(f[-1].item(), 5),
        round(real_idx - 1, 5),
        I0,
    )
