import streamlit as st


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
