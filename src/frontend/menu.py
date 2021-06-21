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
        page_handling.handler.set_page("lol_dataset")
        pass

    st.subheader("Algorithms")
    if st.button("Run algorithms"):
        page_handling.handler.set_page("als")

    # map shot description of dataset attr. to long description to display in menu
    reddit_attribute_mapping = dict(r_users="Min. User per Subreddit", r_comments="Min. Comment per Subreddit",
                                    u_comments="Min. Comment per User", u_reddit="Min. Subreddit per User",
                                    include_over18="Subreddits over 18?", alpha="Alpha")
    lol_attribute_mapping = dict()  # TODO

    if state.datasets:
        st.subheader("Configured dataset:")
        st.markdown("<div class='separator'></div>", unsafe_allow_html=True)
        for i, dataset in enumerate(state.datasets):
            # three cols for 'css hack'
            c1, _, c2 = st.beta_columns([8, 2, 1])

            mapping = reddit_attribute_mapping if dataset.id == 0 else lol_attribute_mapping
            x = "".join(f"<pre>{mapping[i]}: {j}</pre>" for i, j in dataset.parameter.items())

            c1.markdown("""
            <div class='dataset'>
                    <div class='id'>{0}.</div>
                    <div class='name'>{1}</div>
                    <div class='type'>{2}</div> 
                    <span class='details'>{3}</span>
            </div>
                    """.format(i, dataset.name, 'Subreddit' if dataset.id == 0 else 'League of Legends', x),
                        unsafe_allow_html=True)

            if c2.button("❌", key=f"ds_del_{i}"):
                del state.datasets[i]
                force_rerun()
            st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

    st.write("Made by Alexander Leonhardt and Timothy Schaumlöffel.")
