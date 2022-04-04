import random

import numpy as np
import streamlit as st

st.set_page_config(layout="wide")
from interactive.generate import generate_df_from_pokedex, generate_exp_from_prior
from interactive.plots import plotly_results
from solver.ode import System

random.seed(42)
np.random.seed(42)

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.write(
        """# Virus Mutation Simulation
    In this web app you will be able to configure and run various configuration of diseases!"""
    )

st.write("#")

left, center, right = st.columns([1, 0.1, 1])
with left:
    st.write("### Variation Settings")
    dimension = st.slider("Number of different variants to begin with", 1, 5, 1)

    st.write("""Let's now define the main characterstics of those diseases""")

    lamda = st.slider("Infectiousness", 0.0, 1.0, 0.6)
    gamma = st.slider("Recovery Rate", 0.0, 1.0, 0.1)
    alpha = st.slider("Antibodies Loss Rate", 0.0, 1.0, 0.1)
    beta = st.slider("Re-Illness Rate", 0.0, 1.0, 0.1)
    frequency = st.slider("Mutation Likelihood", 0.0, 1.0, 0.0)

    unit_size = st.slider("Outbreak Size %", 0.0, 1.0, 0.1) if frequency else 1e-3


l, g, B, a, f, X0 = generate_exp_from_prior(dimension, lamda, gamma, beta, alpha, frequency)

# Defining the settings for the simulation
steps = 100
lenght = 25
# Solving the simulation
system = System(
    X0, l, g, B, a, f, lenght, steps, mutation=frequency != 0, unit_size=unit_size / 100
)
y, t, pokedex = system.solve()
dimension = round((y.shape[-1] - 1) / 3)


with right:
    st.write("### Simulation Result")
    st.write("#")
    # Plotting results
    options = ["All", *[f"Variant {i+1}" for i in range(dimension)]]
    idx = st.selectbox("Which graph do you want to see?", options)
    idx = options.index(idx)

    healthy = st.checkbox("Plot healthy line", value=True)

    fig = plotly_results(y, t, pokedex, idx, healthy)
    st.plotly_chart(fig)

st.write("#")
st.write("### Pokedex of Variations")

idx = st.selectbox("Select the variant you want to take a closer look at", options)
idx = options.index(idx)

pokedex = generate_df_from_pokedex(pokedex, idx)
st.dataframe(pokedex)
