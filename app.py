import random

import numpy as np
import streamlit as st

st.set_page_config(page_title="Virus Simulator", layout="wide")
from components import show_pokedex, sidebar, title, variant_setting
from components.plots import plotly_results
from solver.ode import System
from solver.register import Variant
from utils.generate import add_variant, build_starting_point, generate_var_from_prior

random.seed(42)
np.random.seed(42)

if "pool" not in st.session_state:
    st.session_state.pool = []

title()
with st.sidebar:
    unit_size, sick_size = sidebar()

st.write("#")
left1, _, left2, center, right = st.columns([0.5, 0.01, 0.5, 0.05, 1])
dimension, lamda, gamma, alpha, beta, frequency, idx_to_plot, susceptible = variant_setting(
    left1, left2
)


vars = generate_var_from_prior(dimension, lamda, gamma, beta, alpha, frequency)
l, g, B, a, f, X0 = build_starting_point(st.session_state.pool or vars, sick_size / 100)
if st.session_state.pool:
    if len(vars) != 1 or vars[0] != st.session_state.pool[-1]:
        for var in vars:
            l, g, B, a, f, X0 = add_variant(
                var, l, g, B, a, f, X0, rebalance=True, sick_size=sick_size / 100
            )


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

    st.write("Here below you can see the interpolated solution, with the given parameters")
    fig = plotly_results(y, t, pokedex, idx_to_plot, susceptible)
    st.plotly_chart(fig)

st.write("#")
show_pokedex(pokedex, dim=dimension)
