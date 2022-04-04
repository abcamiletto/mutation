import numpy as np
import pandas as pd


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
    """Generate random experiments of dimension dim"""
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
