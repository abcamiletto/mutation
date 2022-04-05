import streamlit as st


def title():
    _, col2, _ = st.columns([1.5, 5, 1.5])
    with col2:
        st.write(
            """# Virus Mutation Simulation
        In this web app you will be able to configure and run various configuration of diseases!"""
        )
