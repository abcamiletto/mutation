import pathlib
import sys

import streamlit as st

here = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(here))
from solver.register import Variant


def variant_setting(col1, col2, show_env):

    with col1:
        st.write("### Variation Settings")
        dimension = st.slider("Number of different variants to begin with", 1, 5, 1)

        lamda = st.slider("Infectiousness", 0.0, 1.0, 0.6)
        gamma = st.slider("Recovery Rate", 0.0, 1.0, 0.1)

    with col2:
        alpha = st.slider("Antibodies Loss Rate", 0.0, 1.0, 0.1)
        beta = st.slider("Re-Illness Rate", 0.0, 1.0, 0.1)
        frequency = st.slider("Mutation Likelihood", 0.0, 1.0, 0.0)
        if show_env:
            if st.button("Add to Environment"):
                v = Variant(lamda, gamma, beta, alpha, frequency)
                st.session_state.pool.append(v)

    return dimension, lamda, gamma, alpha, beta, frequency


def env_settings(visible):
    if not visible:
        return 0.1, 1.0

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

    return unit_size, sick_size
