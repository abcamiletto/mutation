import numpy as np
import pandas as pd


def generate_df_from_pokedex(pokedex, idx):
    pokedex = [
        variant.__dict__ | dict(name=f"Variant {idx+1}") for idx, variant in enumerate(pokedex)
    ]

    df_pokedex = pd.DataFrame(pokedex)
    df_pokedex = df_pokedex.set_index("name", drop=True)
    df_pokedex = df_pokedex.T

    if idx != 0:
        df_pokedex = df_pokedex[f"Variant {idx}"]

    return df_pokedex


def generate_exp_from_prior(dim, l, g, B, a, f):
    """Generate random experiments of dimension dim"""
    if dim == 1:
        l = np.ones((1, 1)) * l
        g = np.ones((1, 1)) * g
        a = np.ones((1, 1)) * a
        f = np.ones((1, 1)) * f
        B = np.ones((1, 1)) * B
    # If we have different starting variation we differentiate between them
    else:
        l = l + np.random.rand(dim, 1) / 10
        g = g + np.random.rand(dim, 1) / 10
        a = a + np.random.rand(dim, 1) / 10
        f = f + np.random.rand(dim, 1) / 10
        B = B + np.random.rand(dim, dim) / 10

    I0 = np.ones(dim) / (10 * dim)
    S0 = 1 - I0.sum()
    R0 = [0] * dim
    W0 = [0] * dim

    X0 = np.array([S0, *I0, *R0, *W0])
    return l, g, B, a, f, X0
