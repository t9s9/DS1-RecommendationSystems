import streamlit as st
import src.frontend.page_handling as page_handling


def app():
    if st.button("Reddit dataset"):
        page_handling.handler.set_page("reddit_dataset")