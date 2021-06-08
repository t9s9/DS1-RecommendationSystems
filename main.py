import streamlit as st

from src.data import check_all_datasets
from src.frontend.page_handling import handler

st.set_page_config(page_title="Recommender Systems",
                   page_icon="\U0001F4BE",
                   initial_sidebar_state='auto',
                   layout='centered')

check_all_datasets()

handler.run()
