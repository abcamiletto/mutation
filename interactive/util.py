import pandas as pd


def df_from_pokedex(pokedex, idx):
    pokedex = [
        variant.__dict__ | dict(name=f"Variant {idx+1}") for idx, variant in enumerate(pokedex)
    ]

    df_pokedex = pd.DataFrame(pokedex)
    df_pokedex = df_pokedex.set_index("name", drop=True)
    df_pokedex["parent"] += 1
    df_pokedex = df_pokedex.T

    if idx != 0:
        df_pokedex = df_pokedex[f"Variant {idx}"]

    return df_pokedex
