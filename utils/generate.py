import pathlib
import sys

import numpy as np
import pandas as pd

here = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(here))

from solver.state import pack, unpack
from solver.util import Variant


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


def generate_var_from_prior(dim, l, g, B, a, f, I0):
    """Generate dim variants from priors"""
    vars = []
    rand = np.random.rand(dim, 6) / 10
    if dim > 1:
        for i in range(dim):
            lamda = l + rand[i, 0]
            gamma = g + rand[i, 1]
            beta_self = B + rand[i, 2]
            alpha = a + rand[i, 3]
            freq = f + rand[i, 4]
            i = I0
            vars.append(Variant(lamda, gamma, beta_self, alpha, freq, None, i))
    else:
        vars.append(Variant(l, g, B, a, f, None, I0))
    return vars


def build_starting_point(variants, sick_size=None):
    var = variants[0]
    l = np.expand_dims(np.array(var.lamda), axis=(0, 1))
    g = np.expand_dims(np.array(var.gamma), axis=(0, 1))
    B = np.expand_dims(np.array(var.beta_self), axis=(0, 1))
    a = np.expand_dims(np.array(var.alpha), axis=(0, 1))
    f = np.expand_dims(np.array(var.frequency), axis=(0, 1)).clip(min=1e-6)
    I0 = sick_size or var.I0
    X0 = np.array([1 - I0, I0, 0, 0])
    for var in variants[1:]:
        l, g, B, a, f, X0 = add_variant(var, l, g, B, a, f, X0, sick_size=sick_size, unit=I0)
    return l, g, B, a, f, X0


def add_variant(variant, l, g, B, a, f, X0, sick_size=None, unit=1e-3):
    l = np.concatenate([l, np.full((1, 1), variant.lamda)]).clip(min=0)
    g = np.concatenate([g, np.full((1, 1), variant.gamma)]).clip(min=0)
    a = np.concatenate([a, np.full((1, 1), variant.alpha)]).clip(min=0)
    f = np.concatenate([f, np.full((1, 1), variant.frequency)]).clip(min=1e-6)

    size = B.shape[0]
    new_diagonal = np.concatenate([np.diagonal(B), np.full((1,), variant.beta_self)])
    B = np.ones((size + 1, size + 1)) * 0.2
    np.fill_diagonal(B, new_diagonal)

    S, I, R, W = unpack(X0)

    S = np.expand_dims(S, 1)
    if sick_size:
        I = np.ones(shape=(len(I) + 1, 1)) * sick_size / (len(I) + 1)
    else:
        I = np.expand_dims(np.append(I, unit), 1)
    R = np.expand_dims(np.append(R, 0), 1)
    W = np.expand_dims(np.append(W, 0), 1)

    X0 = pack([S, I, R, W])
    return l, g, B, a, f, X0
