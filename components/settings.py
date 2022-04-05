import pathlib
import sys

import streamlit as st

here = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(here))
from solver.register import Variant


def variant_setting(col1, col2):

    with col1:
        st.write("### Variation Settings")
        dimension = st.slider("Number of different variants to begin with", 1, 5, 1)

        lamda = st.slider("Infectiousness", 0.0, 1.0, 0.6)
        gamma = st.slider("Recovery Rate", 0.0, 1.0, 0.1)

    with col2:
        alpha = st.slider("Antibodies Loss Rate", 0.0, 1.0, 0.1)
        beta = st.slider("Re-Illness Rate", 0.0, 1.0, 0.1)
        frequency = st.slider("Mutation Likelihood", 0.0, 1.0, 0.0)

        if st.button("Add to Environment"):
            v = Variant(lamda, gamma, beta, alpha, frequency)
            st.session_state.pool.append(v)

    return dimension, lamda, gamma, alpha, beta, frequency


def sidebar():

    st.write("### Environment Settings")
    st.write("Set global settings of the environment")

    st.write("")
    sick_size = st.slider("Infected people at the beginning, % of total population", 0.0, 10.0, 1.0)

    st.write("")
    unit_size = st.slider("Outbreak Size, % of total population", 0.0, 1.0, 0.1)

    st.file_uploader("Load from File for a finer control")

    st.write("")

    left, right = st.columns([1, 1])
    with left:
        if st.button("Reset"):
            st.session_state.pool = []
    with right:
        if st.button("Undo") and len(st.session_state.pool) > 1:
            st.session_state.pool.pop()

    return unit_size, sick_size
