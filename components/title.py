import streamlit as st


def title():
    _, col2, _ = st.columns([3, 5, 3])
    with col2:
        st.write(
            """# Virus Mutation Simulation
        In this web app you will be able to configure and run various configuration of diseases!"""
        )
