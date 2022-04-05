import streamlit as st


def title():
    _, col2, _, right = st.columns([3, 5, 2, 1.0])
    with col2:
        st.write(
            """# Virus Mutation Simulation
        In this web app you will be able to configure and run various configuration of diseases!"""
        )
    with right:
        show_env = st.checkbox("PRO mode", value=False)

    return show_env
