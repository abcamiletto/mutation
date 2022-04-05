import random

import numpy as np
import streamlit as st

from utils.storing import load_experiment

st.set_page_config(page_title="Virus Simulator", layout="wide")
import time

from components import show_pokedex, sidebar, title, variant_setting
from components.plots import plotly_results
from solver.ode import System
from utils.generate import add_variant, build_starting_point, generate_var_from_prior

random.seed(42)
np.random.seed(42)

if "pool" not in st.session_state:
    st.session_state.pool = []

#   TITLE
title()

#   SIDEBAR
with st.sidebar:
    sim_lenght, unit_size, sick_size, uploaded_file = sidebar()

#   VARIANT SETTINGS
st.write("#")
_, left, _, right = st.columns([1, 2, 1.3, 2])
with left:
    st.write("### Variation Settings")
with right:
    st.write("### Simulation Results")

left1, _, left2, center, right = st.columns([0.5, 0.01, 0.5, 0.05, 1])
dimension, lamda, gamma, alpha, beta, frequency = variant_setting(left1, left2)

#   GENERATING STARTING POINT
if uploaded_file is not None:
    l, g, B, a, f, X0 = load_experiment(file=uploaded_file)
else:
    vars = generate_var_from_prior(dimension, lamda, gamma, beta, alpha, frequency)
    l, g, B, a, f, X0 = build_starting_point(st.session_state.pool or vars, sick_size / 100)
    if st.session_state.pool:
        if len(vars) != 1 or vars[0] != st.session_state.pool[-1]:
            for var in vars:
                l, g, B, a, f, X0 = add_variant(
                    var, l, g, B, a, f, X0, rebalance=True, sick_size=sick_size / 100
                )


#   SIMULATION SETTINGS
steps = sim_lenght * 4
mutation = bool(st.session_state.pool) or (frequency != 0)

#   SOLVING THE MODEL
tic = time.time()
system = System(X0, l, g, B, a, f, sim_lenght, steps, mutation=mutation, unit_size=unit_size / 100)
y, t, pokedex = system.solve()
toc = time.time() - tic
print(f"Time needed to simulate the model {toc:.3f}s")
dimension = round((y.shape[-1] - 1) / 3)

#   PLOTTING RESULTS
with right:
    options = ["All", *[f"Variant {i+1}" for i in range(dimension)]]
    idx = st.selectbox("Which graph do you want to se?", options)
    idx = options.index(idx)
    susceptible = st.checkbox("Plot susceptible line", value=True)
    fig = plotly_results(y, t, pokedex, idx, susceptible)
    st.plotly_chart(fig)

#   POKEDEX
st.write("#")
show_pokedex(pokedex, dim=dimension)
