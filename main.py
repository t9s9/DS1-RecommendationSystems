import streamlit as st

from src.data import check_all_datasets
from src.frontend import reddit_dataset

st.set_page_config(page_title="Recommender Systems",
                   page_icon="\U0001F4BE",
                   initial_sidebar_state='auto',
                   layout='centered')

check_all_datasets()

# do multi-app management here / Main menu
reddit_dataset.app()
