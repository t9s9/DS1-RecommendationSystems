import streamlit as st

from src.data import check_all_datasets
from src.frontend.SessionState import session_get
from src.frontend.page_handling import handler

st.set_page_config(page_title="Recommender Systems",
                   page_icon="\U0001F4BE",
                   initial_sidebar_state='auto',
                   layout='centered')

check_all_datasets()

# Set default values in Session State
state = session_get(datasets=[],
                    reddit_config=dict(u_comments=20, u_reddit=20, r_comments=100, r_users=100, include_over18=False,
                                       alpha=1, name="Subreddit_dataset_1"))
print("Current session_state:", state.__dict__)
handler.run()
