import pathlib
import sys

import streamlit as st

here = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(here))
from solver.util import Variant


def variant_setting(col1, col2):
    with col1:
        dimension = st.slider("Number of different variants to spawn", 1, 5, 1)

        lamda = st.slider("Infectivity", 0.0, 1.0, 0.6)
        gamma = st.slider("Recovery Rate", 0.0, 1.0, 0.1)

    with col2:
        alpha = st.slider("Immunity Loss Rate", 0.0, 1.0, 0.1)
        beta = st.slider("Reinfection Rate", 0.0, 1.0, 0.1)
        frequency = st.slider("Mutation Rate", 0.0, 1.0, 0.0)

        if st.button("Add to Environment"):
            v = Variant(lamda, gamma, beta, alpha, frequency)
            st.session_state.pool.append(v)

    return dimension, lamda, gamma, alpha, beta, frequency
