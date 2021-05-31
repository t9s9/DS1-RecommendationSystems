import streamlit as st
import reddit_dataset

st.set_page_config(page_title="Recommender Systems",
                   page_icon="\U0001F4BE",
                   initial_sidebar_state='auto',
                   layout='wide')

# do multi-app management here / Main menu

reddit_dataset.app()
