import streamlit as st
import src.frontend.page_handling as page_handling

from src.frontend.SessionState import session_get
from src.frontend.util import timer

from src.algorithm.als.model_wrapper import ImplicitModelWrapper


def als_configuration(datasets):
    dataset_names = [d.name for d in datasets]

    conf = dict()
    conf_widget = st.form(key="inf_conf")
    conf['datasets'] = conf_widget.multiselect(label="Datasets", options=dataset_names)
    conf_widget.markdown("---")
    conf['test_ratio'] = conf_widget.number_input("Test ratio", min_value=0.0, max_value=1.0, value=0.1, step=0.1,
                                                  help="")
    conf_widget.markdown("---")
    conf['iterations'] = conf_widget.number_input("Iterations", min_value=1, value=10, step=1, help="")
    conf['factors'] = conf_widget.number_input("Factors", min_value=1, value=50, step=1, help="")
    conf['regularization'] = conf_widget.number_input("Regularization", value=0.001, step=0.001, format="%.4f", help="")
    submit = conf_widget.form_submit_button("Start")
    if submit:
        conf['datasets_id'] = []
        for i in conf['datasets']:
            conf['datasets_id'].append(dataset_names.index(i))

    return submit, conf


@timer
def app():
    state = session_get()

    if st.button("Back"):
        page_handling.handler.set_page("menu")

    start, config = False, None
    with st.sidebar:
        st.title("Configuration")
        conf_datasets = st.selectbox(label="Algorithm", options=["ALS", "KNN"])
        if conf_datasets == "ALS":
            start, config = als_configuration(state.datasets)

    if start:
        if conf_datasets == "ALS":
            for i in config['datasets_id']:
                print("Running als on", state.datasets[i])
                model = ImplicitModelWrapper(dataset=state.datasets[i], iterations=config['iterations'],
                                             factors=config['factors'], regularization=config['regularization']).fit()


    st.subheader("Recommend the top N items to an user")
    col1, col2 = st.beta_columns([2, 1])
    col1.text_input("User", help="Select a user for which item recommendations are to be made.")
    col2.number_input("N", help="The number of top recommendations", min_value=1, step=1, value=10, key="N_users")

    st.subheader("Find the top N similar item to an item")
    col1, col2 = st.beta_columns([2, 1])
    col1.text_input("Item", help="Select an item to find the most similar other items.")
    col2.number_input("N", help="The number of top similar items", min_value=1, step=1, value=10, key="N_items")
