import pandas as pd
import streamlit as st
import src.frontend.page_handling as page_handling
from src.frontend.SessionState import session_get
from src.frontend.util import force_rerun
from src.frontend.util import timer

@timer
def app():
    state = session_get()

    st.write(f"<style>{open('src/frontend/css/menu.css').read()}</style>", unsafe_allow_html=True)

    st.title("Menu")
    st.markdown("1. Select and configure datasets.\n2. Run and evaluate on different algorithms.")

    st.subheader("Datasets")
    q1, q2 = st.beta_columns(2)
    if q1.button("Subreddit"):
        page_handling.handler.set_page("reddit_dataset")
    if q2.button("League of Legends"):
        pass

    st.subheader("Algorithms")
    if st.button("Run algorithms"):
        page_handling.handler.set_page("als")

    if state.datasets:
        st.subheader("Configured dataset:")
        for i, dataset in enumerate(state.datasets):
            print(dataset.name, dataset.parameter)
            # three cols for 'css hack'
            c1, _, c2 = st.beta_columns([9, 1, 1])
            # TODO: more beauty
            c1.markdown("""
            <div class='dataset'>
                <div class='name'>{0}. {1}</div>
                <div class='parameter'>{2}</div>
            </div>""".format(i, dataset.name, dataset.parameter), unsafe_allow_html=True)
            if c2.button("❌", key=f"ds_del_{i}"):
                del state.datasets[i]
                force_rerun()

    st.write("Made by Alexander Leonhardt and Timothy Schaumlöffel.")
