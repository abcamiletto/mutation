import pandas as pd
import streamlit as st


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


def show_pokedex(pokedex, dim):
    st.write("### Pokedex of Variations")
    options = ["All", *[f"Variant {i+1}" for i in range(dim)]]
    idx = st.selectbox("Select the variant you want to take a closer look at", options)
    idx = options.index(idx)

    pokedex = df_from_pokedex(pokedex, idx)
    st.dataframe(pokedex, height=400)
