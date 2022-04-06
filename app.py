import random

import numpy as np
import streamlit as st

from utils.storing import load_experiment, save_experiment

st.set_page_config(page_title="Virus Simulator", layout="wide")
import time

from components import show_pokedex, sidebar, title, variant_setting
from components.plots import plotly_results
from solver.ode import System
from utils.generate import add_variant, build_starting_point, generate_from_prior

random.seed(42)
np.random.seed(42)

if "pool" not in st.session_state:
    st.session_state.pool = []

#   TITLE
title()

#   SIDEBAR
with st.sidebar:
    sim_lenght, unit_size, sick_size, uploaded_file, override_I0 = sidebar()

#   VARIANT SETTINGS
st.write("#")
_, left, _, right = st.columns([1, 2, 1.3, 2])
with left:
    st.write("### Variation Settings")
with right:
    st.write("### Simulation Results")

left1, _, left2, center, right = st.columns([0.5, 0.01, 0.5, 0.05, 1])
user_variants = variant_setting(left1, left2)

#   GENERATING STARTING POINT
if uploaded_file is not None:
    starting_point = load_experiment(file=uploaded_file)
else:
    # Building a starting point based on the pool (if any) or the current user defined variant
    starting_point = build_starting_point(
        variants=st.session_state.pool or user_variants,
        sick_size=sick_size / 100 if override_I0 else None,
    )
# If we have a pool of variants saved, we then add the one currently defined by the user
# The same if instead of a pool of variants we have uplaoded an experiment
if st.session_state.pool or uploaded_file:
    if uploaded_file or (user_variants[0] != st.session_state.pool[-1]):
        for var in user_variants:
            starting_point = add_variant(
                var,
                *starting_point,
                sick_size=sick_size / 100 if override_I0 else None,
                unit=var.I0,  # overriden by sick_size
            )

#   DISPLAYING INFO ABOUT SAVED VARIANTS
pool_lenght = len(st.session_state.pool)
if pool_lenght:
    with left1:
        st.write("#")
        st.markdown(f"  * **Variants** saved in the pool : {''.join(['I '] * pool_lenght)}")


#   SIMULATION SETTINGS
steps = sim_lenght * 4
mutation = bool(st.session_state.pool) or (user_variants[0].frequency != 0) or (uploaded_file)

tic = time.perf_counter()
#   SOLVING THE MODEL
l, g, B, a, f, X0 = starting_point
system = System(X0, l, g, B, a, f, sim_lenght, steps, mutation=mutation, unit_size=unit_size / 100)
y, t, pokedex = system.solve()
dimension = round((y.shape[-1] - 1) / 3)

toc = time.perf_counter() - tic
print(f"Time needed to simulate the model {toc:.3f}s")

with st.sidebar:
    st.write("#### Save Experiment ")
    file = save_experiment(l, g, B, a, f, X0, returns=True)
    _, center, _ = st.columns([1, 5, 1])
    with center:
        st.download_button("Download Current Experiment", file, file_name="current_exp.yaml")


#   PLOTTING RESULTS
with right:
    options = ["All", *[f"Variant {i+1}" for i in range(dimension)]]
    idx = st.selectbox("Which graph do you want to see?", options)
    idx = options.index(idx)
    susceptible = st.checkbox("Plot susceptible line", value=True)
    tic = time.perf_counter()
    fig = plotly_results(y, t, pokedex, idx, susceptible)
    toc = time.perf_counter() - tic
    print(f"Time needed to plot the results {toc:.3f}s")
    st.plotly_chart(fig)

#   POKEDEX
show_pokedex(pokedex, dim=dimension)
