import random

import numpy as np
import streamlit as st

st.set_page_config(layout="wide")
from interactive.plots import plotly_results
from interactive.util import df_from_pokedex
from solver.ode import System
from utils.generate import generate_exp_from_prior

random.seed(42)
np.random.seed(42)

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.write(
        """# Virus Mutation Simulation
    In this web app you will be able to configure and run various configuration of diseases!"""
    )

st.write("#")

left1, left2, center, right = st.columns([0.5, 0.5, 0.05, 1])
with left1:
    st.write("### Variation Settings")
    dimension = st.slider("Number of different variants to begin with", 1, 5, 1)

    lamda = st.slider("Infectiousness", 0.0, 1.0, 0.6)
    gamma = st.slider("Recovery Rate", 0.0, 1.0, 0.1)
    alpha = st.slider("Antibodies Loss Rate", 0.0, 1.0, 0.1)

with left2:
    beta = st.slider("Re-Illness Rate", 0.0, 1.0, 0.1)
    frequency = st.slider("Mutation Likelihood", 0.0, 1.0, 0.0)

    unit_size = st.slider("Outbreak Size, % of total population", 0.0, 1.0, 0.1)
    sick_size = st.slider("Infected people at the beginning, % of total population", 0.0, 10.0, 0.1)


l, g, B, a, f, X0 = generate_exp_from_prior(
    dimension, lamda, gamma, beta, alpha, frequency, sick_size / 100
)

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
    # Plotting results
    options = ["All", *[f"Variant {i+1}" for i in range(dimension)]]
    idx = st.selectbox("Which graph do you want to see?", options)
    idx = options.index(idx)

    healthy = st.checkbox("Plot healthy line", value=True)

    fig = plotly_results(y, t, pokedex, idx, healthy)
    st.plotly_chart(fig)


st.write("### Pokedex of Variations")

idx = st.selectbox("Select the variant you want to take a closer look at", options)
idx = options.index(idx)

pokedex = df_from_pokedex(pokedex, idx)
st.dataframe(pokedex)
