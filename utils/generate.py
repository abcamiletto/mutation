import pathlib
import sys
from random import random

import numpy as np
import pandas as pd
from numpy.random import normal

here = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(here))

from solver.params_util import BETA_RED_DIAGONAL, NOISE_STD, Variant, augment_beta
from solver.state import pack, unpack


def generate_random_exp(dim, sick_size=0.1):
    """Generate random experiments of dimension dim"""
    l = np.random.rand(dim, 1)
    g = np.random.rand(dim, 1)
    a = np.random.rand(dim, 1)
    f = np.random.rand(dim, 1).clip(min=1e-6)
    B = np.random.rand(dim, dim) / 100
    D = np.random.rand(dim, 3) / 100

    I0 = np.ones(dim) * sick_size / (dim)
    S0 = 1 - I0.sum()
    R0 = [0] * dim
    W0 = [0] * dim

    X0 = np.array([S0, *I0, *R0, *W0])
    return l, g, B, a, f, D, X0


def clip_list(args, min, max):
    return [arg.clip(min=min, max=max) for arg in args]


def generate_from_prior(dim, prior, clipped=False, realistic_clipping=False):
    """Generate dim variants from prior"""
    vars = []
    rand = normal(size=(dim, 6)) * NOISE_STD
    if dim > 1:
        for i in range(dim):
            lamda = prior.lamda + rand[i, 0]
            gamma = prior.gamma + rand[i, 1]
            beta_self = prior.beta_self + rand[i, 2]
            beta_self = beta_self if (beta_self < lamda) else lamda
            alpha = prior.alpha + rand[i, 3]
            freq = prior.frequency + rand[i, 4] if prior.frequency != 0 else np.array(0.0)
            death = prior.dI + rand[i, 5]
            i = prior.I0

            # Clipping values
            if clipped:
                attr = [lamda, gamma, beta_self, alpha, freq, death]
                lamda, gamma, beta_self, alpha, freq, death = clip_list(attr, 0, 1)
            if realistic_clipping:
                gamma = gamma.clip(min=0.1)
                alpha = alpha.clip(min=0.1)

            vars.append(Variant(lamda, gamma, beta_self, alpha, freq, death, 0, 0, None, i))
    else:
        vars.append(prior)
    return vars


def build_starting_point(variants, sick_size=None):
    var = variants[0]
    l = np.expand_dims(np.array(var.lamda), axis=(0, 1))
    g = np.expand_dims(np.array(var.gamma), axis=(0, 1))
    B = np.expand_dims(np.array(var.beta_self) / BETA_RED_DIAGONAL, axis=(0, 1))
    a = np.expand_dims(np.array(var.alpha), axis=(0, 1))
    f = np.expand_dims(np.array(var.frequency), axis=(0, 1)).clip(min=1e-6)
    D = np.expand_dims(np.array([var.dI, var.dR, var.dW]), axis=0)

    I0 = sick_size or var.I0
    X0 = np.array([1 - I0, I0, 0, 0])
    for var in variants[1:]:
        l, g, B, a, f, D, X0 = add_variant(
            var, l, g, B, a, f, D, X0, sick_size=sick_size, unit=var.I0
        )
    return l, g, B, a, f, D, X0


def add_variant(variant, l, g, B, a, f, D, X0, sick_size=None, unit=1e-3):
    l = np.concatenate([l, np.full((1, 1), variant.lamda)]).clip(min=0)
    g = np.concatenate([g, np.full((1, 1), variant.gamma)]).clip(min=0)
    a = np.concatenate([a, np.full((1, 1), variant.alpha)]).clip(min=0)
    f = np.concatenate([f, np.full((1, 1), variant.frequency)]).clip(min=1e-6)

    B = augment_beta(B, variant.beta_self)
    D = np.concatenate([D, np.array([variant.dI, variant.dR, variant.dW])[None, ...]]).clip(min=0)
    S, I, R, W = unpack(X0)

    if sick_size:
        S = np.expand_dims(S, 1)
        I = np.ones(shape=(len(I) + 1, 1)) * sick_size / (len(I) + 1)
    else:
        S = np.expand_dims(S - unit, 1)
        I = np.expand_dims(np.append(I, unit), 1)
    R = np.expand_dims(np.append(R, 0), 1)
    W = np.expand_dims(np.append(W, 0), 1)
    X0 = pack([S, I, R, W])
    return l, g, B, a, f, D, X0


def generate_random_vars(dim, sick_size=0.1, mutation=True):
    """Generate random dim variants"""
    vars = []
    for i in range(dim):
        lamda = random()
        gamma = random()
        beta_self = random()
        beta_self = beta_self if (beta_self < lamda) else lamda
        alpha = random()
        frequency = random() if mutation else 0
        dI = random()
        I0 = sick_size / dim
        var = Variant(lamda, gamma, beta_self, alpha, frequency, dI, I0=I0)
        vars.append(var)

    return vars
