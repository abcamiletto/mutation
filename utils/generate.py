import pathlib
import sys

import numpy as np
import pandas as pd

here = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(here))

from solver.model import pack, unpack


def generate_random_exp(dim, sick_size=0.1):
    """Generate random experiments of dimension dim"""
    l = np.random.rand(dim, 1)
    g = np.random.rand(dim, 1)
    a = np.random.rand(dim, 1)
    f = np.random.rand(dim, 1).clip(min=1e-6)
    B = np.random.rand(dim, dim) / 100

    I0 = np.ones(dim) * sick_size / (dim)
    S0 = 1 - I0.sum()
    R0 = [0] * dim
    W0 = [0] * dim

    X0 = np.array([S0, *I0, *R0, *W0])
    return l, g, B, a, f, X0


def generate_exp_from_prior(dim, l, g, B, a, f, sick_size=0.1):
    """Generate random experiments of dimension dim from priors"""
    if dim == 1:
        l = np.ones((1, 1)) * l
        g = np.ones((1, 1)) * g
        a = np.ones((1, 1)) * a
        f = (np.ones((1, 1)) * f).clip(min=1e-6)
        B = np.ones((1, 1)) * B
    # If we have different starting variation we differentiate between them
    else:
        l = l + np.random.rand(dim, 1) / 10
        g = g + np.random.rand(dim, 1) / 10
        a = a + np.random.rand(dim, 1) / 10
        f = f + np.random.rand(dim, 1) / 10
        B = B + np.random.rand(dim, dim) / 10

    I0 = np.ones(dim) * sick_size / (dim)
    S0 = 1 - I0.sum()
    R0 = [0] * dim
    W0 = [0] * dim

    X0 = np.array([S0, *I0, *R0, *W0])
    return l, g, B, a, f, X0


def build_starting_point(variants, sick_size):
    var = variants[0]
    l = np.expand_dims(np.array(var.lamda), axis=(0, 1))
    g = np.expand_dims(np.array(var.gamma), axis=(0, 1))
    B = np.expand_dims(np.array(var.beta_self), axis=(0, 1))
    a = np.expand_dims(np.array(var.alpha), axis=(0, 1))
    f = np.expand_dims(np.array(var.frequency), axis=(0, 1)).clip(min=1e-6)
    X0 = np.array([1 - sick_size, sick_size, 0, 0])
    for var in variants[1:]:
        l, g, B, a, f, X0 = add_variant(var, l, g, B, a, f, X0, rebalance=True)
    return l, g, B, a, f, X0


def add_variant(variant, l, g, B, a, f, X0, rebalance=False, sick_size=0.1, unit=1e-3):
    l = np.concatenate([l, np.full((1, 1), variant.lamda)]).clip(min=0)
    g = np.concatenate([g, np.full((1, 1), variant.gamma)]).clip(min=0)
    a = np.concatenate([a, np.full((1, 1), variant.alpha)]).clip(min=0)
    f = np.concatenate([f, np.full((1, 1), variant.frequency)]).clip(min=1e-6)
    B = np.diag(np.concatenate([np.diagonal(B), np.full((1,), variant.beta_self)])).clip(min=0)

    S, I, R, W = unpack(X0)

    S = np.expand_dims(S, 1)
    if rebalance:
        I = np.ones(shape=(len(I) + 1, 1)) * sick_size / (len(I) + 1)
    else:
        I = np.expand_dims(np.append(I, unit), 1)
    R = np.expand_dims(np.append(R, 0), 1)
    W = np.expand_dims(np.append(W, 0), 1)

    X0 = pack([S, I, R, W])
    return l, g, B, a, f, X0
