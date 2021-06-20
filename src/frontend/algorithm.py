import pandas as pd
import streamlit as st
import src.frontend.page_handling as page_handling

from src.frontend.SessionState import session_get
from src.frontend.util import timer
import threading

from src.algorithm.als.model_wrapper import ImplicitModelWrapper


def als_configuration(datasets):
    dataset_names = [d.name for d in datasets]

    conf = dict()
    conf_widget = st.form(key="inf_conf")
    conf['datasets'] = conf_widget.multiselect(label="Datasets", options=dataset_names)
    conf_widget.markdown("---")
    conf['iterations'] = conf_widget.number_input("Iterations", min_value=1, value=10, step=1, help="")
    conf['factors'] = conf_widget.number_input("Factors", min_value=1, value=50, step=1, help="")
    conf['regularization'] = conf_widget.number_input("Regularization", value=0.001, step=0.001, format="%.4f", help="")
    conf_widget.markdown("---")
    conf['test_ratio'] = conf_widget.number_input("Test ratio", min_value=0.0, max_value=1.0, value=0.1, step=0.1,
                                                  help="")
    conf['metric'] = conf_widget.selectbox("Metric", options=["map"])
    conf['metric_k'] = conf_widget.number_input("Top k", min_value=1, step=1, value=10, help="")

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

    st.title("Algorithm")

    progress_bar = st.progress(0)
    progress_text = st.empty()

    start, config = False, None
    with st.sidebar:
        st.title("Configuration")
        conf_datasets = st.selectbox(label="Algorithm", options=["ALS", "KNN"])
        if conf_datasets == "ALS":
            start, config = als_configuration(state.datasets)

    if start:
        if conf_datasets == "ALS":
            for i in config['datasets_id']:
                this_dataset = state.datasets[i]
                progress_text.write("Creating dataset... (dataframe -> sparse item-user matrix)")
                model = ImplicitModelWrapper(dataset=this_dataset, iterations=config['iterations'],
                                             factors=config['factors'], regularization=config['regularization'],
                                             test_size=config['test_ratio'])

                running = True

                def update_progressbar():
                    while running:
                        val = int((model.current_iteration + 1) / model.iterations * 100)
                        progress_bar.progress(val)
                        progress_text.write(
                            "Fitting dataset {2}:      {0:04d} / {1:04d}  ({3:.3f}s)".format(
                                model.current_iteration + 1,
                                model.iterations, this_dataset.name,
                                model.current_iteration_time))

                progress_updater = threading.Thread(target=update_progressbar)
                st.report_thread.add_report_ctx(progress_updater)
                progress_updater.start()

                model.fit()
                running = False
                progress_updater.join()

                model_data = model.export()
                progress_text.write("Evaluating...")
                model_data['evaluation'] = model.evaluate(metric=config['metric'], k=config['metric_k'])
                this_dataset.trainings_als.append(model_data)
                progress_text.write("")
                progress_bar.progress(0)

    als_result_df = []
    for dataset in state.datasets:
        for i, training in enumerate(dataset.trainings_als):
            d = dict(tid=i + 1, name=dataset.name)
            d.update(training['config'])
            d.update(training['evaluation'])
            als_result_df.append(d)
    if als_result_df:
        st.header("Training results")
        st.subheader("ALS")
        als_result_df = pd.DataFrame(als_result_df).set_index(['tid', 'name'])

        def highlight_max(s):
            return ["background-color : red"]

        als_result_df.style.apply(highlight_max)
        st.write(als_result_df)

        with st.beta_expander("Choose a trained model for inference:"):
            active_training = st.selectbox("Current training", options=als_result_df.index.values,
                                           format_func=lambda x: f"Training: {x[0]} - Dataset: {x[1]}")

            if active_training:
                # find dataset
                dataset_id = [d.name for d in state.datasets].index(active_training[1])
                this_dataset = state.datasets[dataset_id]
                this_training = this_dataset.trainings_als[active_training[0] - 1]

                inference_model = ImplicitModelWrapper(this_dataset,
                                                       factors=this_training['config']['factors']).as_inference(
                    user_factors=this_training['user_factors'],
                    item_factors=this_training['item_factors'])

                st.subheader("Recommend the top N items to an user")
                col1, col2 = st.beta_columns([2, 1])
                recommend_user = col1.selectbox("User", options=inference_model.dataset_train.users,
                                                help="Select a user for which item recommendations are to be made.", )
                recommend_n = col2.number_input("N", help="The number of top recommendations", min_value=1, step=1,
                                                value=10,
                                                key="N_users")
                if recommend_user:
                    col1, col2 = st.beta_columns([1, 1])
                    col1.markdown("Recommendations")
                    col1.write(inference_model.recommend(user=recommend_user, N=recommend_n, as_df=True))
                    col2.markdown("True ratings")
                    col2.write(inference_model.get_user_ratings(user=recommend_user, as_df=True))

                st.subheader("Find the top N similar item to an item")
                col1, col2 = st.beta_columns([2, 1])
                similar_item = col1.selectbox("Item", options=inference_model.dataset_train.items,
                                              help="Select an item to find the most similar other items.")
                similar_n = col2.number_input("N", min_value=1, step=1, value=10, key="N_items",
                                              help="The number of top similar items")

                if similar_item:
                    st.write(inference_model.similar_items(similar_item, similar_n, use_df=True))
