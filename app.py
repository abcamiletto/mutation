import random

import numpy as np
import streamlit as st

st.set_page_config(page_title="Virus Simulator", layout="wide")
from interactive.plots import plotly_results
from interactive.util import df_from_pokedex
from solver.ode import System
from solver.register import Variant
from utils.generate import add_variant, build_starting_point, generate_exp_from_prior

random.seed(42)
np.random.seed(42)

if "pool" not in st.session_state:
    st.session_state.pool = []

col1, col2, col3, right = st.columns([3, 5, 2, 1])
with col2:
    st.write(
        """# Virus Mutation Simulation
    In this web app you will be able to configure and run various configuration of diseases!"""
    )
with right:
    env_settings = st.checkbox("Show Env Settings", value=False)

if env_settings:
    col1, col2, col3, col4, col5 = st.columns([3, 4, 4, 3, 0.6])
    with col1:
        st.write("### Environment Settings")
        st.write("Set global settings of the environment")
    with col2:
        st.write("")
        sick_size = st.slider(
            "Infected people at the beginning, % of total population", 0.0, 10.0, 1.0
        )
    with col3:
        st.write("")
        unit_size = st.slider("Outbreak Size, % of total population", 0.0, 1.0, 0.1)
    with col4:
        st.file_uploader("Load from File for a finer control")
    with col5:
        st.write("")
        if st.button("Reset"):
            st.session_state.pool = []
        if st.button("Undo") and len(st.session_state.pool) > 1:
            st.session_state.pool.pop()


else:
    unit_size = 0.1
    sick_size = 1.0

st.write("#")
left1, _, left2, center, right = st.columns([0.5, 0.01, 0.5, 0.05, 1])
with left1:
    st.write("### Variation Settings")
    dimension = st.slider("Number of different variants to begin with", 1, 5, 1)

    lamda = st.slider("Infectiousness", 0.0, 1.0, 0.6)
    gamma = st.slider("Recovery Rate", 0.0, 1.0, 0.1)


with left2:
    alpha = st.slider("Antibodies Loss Rate", 0.0, 1.0, 0.1)
    beta = st.slider("Re-Illness Rate", 0.0, 1.0, 0.1)
    frequency = st.slider("Mutation Likelihood", 0.0, 1.0, 0.0)
    if env_settings:
        if st.button("Add to Environment"):
            v = Variant(lamda, gamma, beta, alpha, frequency)
            st.session_state.pool.append(v)


l, g, B, a, f, X0 = generate_exp_from_prior(
    dimension, lamda, gamma, beta, alpha, frequency, sick_size / 100
)

if st.session_state.pool:
    l, g, B, a, f, X0 = build_starting_point(st.session_state.pool, sick_size / 100)
    var = Variant(lamda, gamma, beta, alpha, frequency)
    if var != st.session_state.pool[-1]:
        l, g, B, a, f, X0 = add_variant(var, l, g, B, a, f, X0, rebalance=True)

# Defining the settings for the simulation
steps = 100
lenght = 25
# Solving the simulation
mutation = bool(st.session_state.pool) or (frequency != 0)
system = System(X0, l, g, B, a, f, lenght, steps, mutation=mutation, unit_size=unit_size / 100)
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
