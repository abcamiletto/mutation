import streamlit as st
import yaml


def sidebar():

    st.write("### Environment Settings")
    st.write("Set global settings of the environment")

    st.write("")
    sim_lenght = st.slider("Lenght of the simulation in weeks", 0, 100, 25)

    st.write("")
    sick_size = st.slider("Infected people at the start, % of total population", 0.0, 10.0, 1.0)

    st.write("")
    unit_size = st.slider("Outbreak Size, % of total population", 0.0, 1.0, 0.1)

    uploaded_file = st.file_uploader("Load from File for a finer control")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()

        uploaded_file = yaml.load(bytes_data, Loader=yaml.FullLoader)

    st.write("")

    left, right = st.columns([1, 1])
    with left:
        if st.button("Reset"):
            st.session_state.pool = []
    with right:
        if st.button("Undo") and len(st.session_state.pool) > 1:
            st.session_state.pool.pop()

    return sim_lenght, unit_size, sick_size, uploaded_file
