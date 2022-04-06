import pathlib
import sys

import streamlit as st

here = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(here))
from solver.params_util import Variant
from utils.generate import generate_from_prior


def variant_setting(col1, col2):
    with col1:
        dimension = st.slider("Number of different variants to spawn", 1, 5, 1)

        lamda = st.slider("Infectivity", 0.0, 1.0, 0.6)
        gamma = st.slider("Recovery Rate", 0.0, 1.0, 0.1)
        alpha = st.slider("Immunity Loss Rate", 0.0, 1.0, 0.1)

    with col2:
        I0 = st.slider("Starting Outbreak Size, % of total population", 0.0, 10.0, 1.0) / 100
        beta = st.slider("Reinfection Rate", 0.0, 1.0, 0.1)
        frequency = st.slider("Mutation Rate", 0.0, 1.0, 0.0)

        variant = Variant(lamda, gamma, beta, alpha, frequency, None, I0)

        user_variants = generate_from_prior(dimension, variant)

        st.write("#")
        if st.button("Add to the Pool"):
            st.session_state.pool.extend(user_variants)

    # Creating a list of the variants currently defined by the user

    return user_variants
