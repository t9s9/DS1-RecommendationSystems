import threading

import pandas as pd
import streamlit as st

import src.frontend.page_handling as page_handling
from src.algorithm.als.model_wrapper import ImplicitModelWrapper
from src.algorithm.knn.model_wrapper import KNNModelWrapper
from src.frontend.SessionState import session_get
from src.frontend.util import timer


def als_configuration(datasets):
    dataset_names = [d.name for d in datasets]

    conf = dict()
    conf_widget = st.form(key="inf_conf")
    conf['datasets'] = conf_widget.multiselect(label="Datasets", options=dataset_names,
                                               help="Choose multiple datasets to run on ALS")
    conf_widget.markdown("---")
    conf['iterations'] = conf_widget.number_input("Iterations", min_value=1, value=10, step=1,
                                                  help="The number of iterations. In each iteration the user and item "
                                                       "factor were calculated separately.")
    conf['factors'] = conf_widget.number_input("Factors", min_value=1, value=64, step=1,
                                               help="The number of latent factors f to use in the matrix decomposition.")
    conf['regularization'] = conf_widget.number_input("Regularization", value=0.1, step=0.001, format="%.4f",
                                                      help="The regularization strength lambda.")
    conf_widget.markdown("---")
    conf['test_ratio'] = conf_widget.number_input("Test ratio", min_value=0.0, max_value=1.0, value=0.1, step=0.1,
                                                  help="The proportion of the dataset that should be used for testing. "
                                                       "It have to be in [0, 1). If set to zero the no evaluation "
                                                       "is performed.")
    conf['metric'] = conf_widget.selectbox("Metric", options=["map", "precision", "auc"],
                                           help="The metric used for evaluation: Mean Average Precision, Precision "
                                                "and Area under the curve")
    conf['metric_k'] = conf_widget.number_input("Top k", min_value=1, step=1, value=10,
                                                help="Number of top values to include into the evaluation")
    conf['cross_validation_folds'] = conf_widget.number_input("Cross validation folds", min_value=0, step=1, value=0,
                                                              help="Perform a cross validation with the given number "
                                                                   "of folds. The test ratio is ignored. If set to zero"
                                                                   " no cross validation is performed.")
    submit = conf_widget.form_submit_button("Start")
    if submit:
        if conf['cross_validation_folds'] > 1:
            conf['test_ratio'] = 0.0
        elif conf['cross_validation_folds'] == 1:
            st.warning("Cross validation with one fold is not possible. Test ratio will be taken.")

        conf['datasets_id'] = []
        for i in conf['datasets']:
            conf['datasets_id'].append(dataset_names.index(i))

    return submit, conf


def highlight_max(s):
    return ["background-color : red"]


def knn_configuration(datasets):
    dataset_names = [d.name for d in datasets]
    form = st.form(key="knn_conf")
    conf = {}
    with form:
        conf['datasets'] = st.multiselect(label="Datasets", options=dataset_names,
                                          help="Choose multiple datasets to run them on KNN")
        st.markdown("---")
        conf["cluster_k"] = st.slider(label="Select value k", min_value=0, max_value=100, value=10,
                                      help="The number of neighbors to take into account")
        conf["similarity"] = st.selectbox(label="Select similarity metric",
                                          options=["cosine", "msd", "pearson", "pearson_baseline"],
                                          help="Specify a distance metric for KNN")
        st.markdown("---")
        conf["metric"] = st.selectbox("Metric", options=["mae", "mse", "ap"],
                                      help="The metric used for evaluation: Mean average error, Mean squared error,"
                                           "and average precision")

        conf['thresh_metric'] = st.number_input("Threshold for recommendation", min_value=0.0, max_value=100.0,
                                                value=0.1, step=0.1,
                                                help="The threshold for the rating during evaluation")
        conf['metric_k'] = st.number_input("Top k (only used when metric is map)", min_value=1, step=1, value=10,
                                           help="Number of top values to include into the evaluation")

        conf['cross_validation_folds'] = st.number_input("Cross validation folds", min_value=2, step=1, value=10,
                                                         help="Perform a cross validation with the given number of "
                                                              "folds.")

        submit = st.form_submit_button("Start")
        if submit:
            conf['datasets_id'] = []
            for i in conf['datasets']:
                conf['datasets_id'].append(dataset_names.index(i))
    return submit, conf


def run_knn(datasets):
    selects = st.sidebar.multiselect("Select the items you have already bought", ["a", "b"])
    val = st.sidebar.slider(label="Select min sample size for calculation", min_value=0, max_value=20, value=10)

    for x in datasets:
        st.write(x)


@timer
def app():
    state = session_get()

    if st.sidebar.button("Back"):
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
        elif conf_datasets == "KNN":
            start, config = knn_configuration(state.datasets)

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
                            "Fitting dataset '{2}'{4}:      {0:04d} / {1:04d}  ({3:.3f}s)".format(
                                model.current_iteration + 1,
                                model.iterations, this_dataset.name,
                                model.current_iteration_time,
                                f"    [Fold: {model.current_fold}]" if config['cross_validation_folds'] > 1 else ""))

                progress_updater = threading.Thread(target=update_progressbar)
                st.report_thread.add_report_ctx(progress_updater)
                progress_updater.start()

                if config['cross_validation_folds'] > 1:
                    r = model.cross_validate(cv=config['cross_validation_folds'], metric=config['metric'],
                                             k=config['metric_k'])
                else:
                    model.fit()
                running = False
                progress_updater.join()

                model_data = model.export()
                if config['cross_validation_folds'] > 1:
                    model_data['test_ratio'] = f"{config['cross_validation_folds']}Fold-CV"
                    model_data['evaluation'] = r
                else:
                    model_data['test_ratio'] = round(config['test_ratio'], 2)
                    if this_dataset.sparse_data_test is not None:
                        progress_text.write("Evaluating...")
                        model_data['evaluation'] = model.evaluate(metric=config['metric'], k=config['metric_k'])
                    else:
                        model_data['evaluation'] = dict(score="", metric="", k="")
                        st.info("An evaluation cannot be performed because no test set was specified.")

                this_dataset.trainings_als.append(model_data)
                progress_text.write("")
                progress_bar.progress(0)
        elif conf_datasets == "KNN":
            for i in config['datasets_id']:
                this_dataset = state.datasets[i]
                model = KNNModelWrapper(dataset=this_dataset, k=config["cluster_k"], sim=config["similarity"])

                model.fit()
                running = False

                model_data = model.export()
                model_data['evaluation'] = model.evaluate(metric=config['metric'], k=config["metric_k"],
                                                          cross_validation_folds=config['cross_validation_folds'],
                                                          thresh=config['thresh_metric'])
                this_dataset.trainings_knn.append(model_data)
                progress_bar.progress(0)

    als_result_df = []
    knn_result_df = []
    for dataset in state.datasets:
        for i, training in enumerate(dataset.trainings_als):
            d = dict(tid=i + 1, name=dataset.name)
            if 'evaluation' in training.keys():
                d.update(training['evaluation'])
                d['test_ratio'] = training['test_ratio']
            d.update(training['config'])
            als_result_df.append(d)
        for i, training in enumerate(dataset.trainings_knn):
            d = dict(tid=i + 1, name=dataset.name)
            d.update(training['config'])
            d.update(training['evaluation'])
            knn_result_df.append(d)

    if als_result_df:
        st.header("Training results")
        st.subheader("ALS")
        als_result_df = pd.DataFrame(als_result_df).set_index(['tid', 'name'])
        als_result_df.style.apply(highlight_max)
        als_result_df = als_result_df
        st.write(als_result_df)
        print(als_result_df)

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
                    col2.markdown("True ratings train set")
                    col2.write(inference_model.get_user_ratings(user=recommend_user, as_df=True))
                    if this_dataset.sparse_data_test is not None:
                        col2.markdown("True ratings test set")
                        col2.write(inference_model.get_user_ratings_test(user=recommend_user, as_df=True))

                st.subheader("Find the top N similar item to an item")
                col1, col2 = st.beta_columns([2, 1])
                similar_item = col1.selectbox("Item", options=inference_model.dataset_train.items,
                                              help="Select an item to find the most similar other items.")
                similar_n = col2.number_input("N", min_value=1, step=1, value=10, key="N_items",
                                              help="The number of top similar items")

                if similar_item:
                    st.write(inference_model.similar_items(similar_item, similar_n, use_df=True))
    if knn_result_df:
        st.header("Training results")
        st.subheader("KNN")
        knn_result_df = pd.DataFrame(knn_result_df).set_index(['tid', 'name'])

        knn_result_df.style.apply(highlight_max)
        st.write(knn_result_df)
        print(knn_result_df)

        with st.beta_expander("Choose a trained model for inference:"):
            active_training = st.selectbox("Current training", options=knn_result_df.index.values, key="unique",
                                           format_func=lambda x: f"Training: {x[0]} - Dataset: {x[1]}")

            if active_training:
                # find dataset
                dataset_id = [d.name for d in state.datasets].index(active_training[1])
                this_dataset = state.datasets[dataset_id]
                this_training = this_dataset.trainings_knn[active_training[0] - 1]

                inference_model = KNNModelWrapper(this_dataset).derive_from(this_training)

                st.subheader("Recommend the top N items to an user")
                col1, col2 = st.beta_columns([2, 1])
                opts = [inference_model.dataset_train.to_raw_uid(x) for x in inference_model.dataset_train.all_users()]
                recommend_user = col1.selectbox("User", options=opts,
                                                help="Select a user for which item recommendations are to be made.",
                                                key="Nuser1")
                recommend_n = col2.number_input("N", help="The number of top recommendations", min_value=1, step=1,
                                                value=10,
                                                key="N_users2")
                if recommend_user:
                    col1, col2 = st.beta_columns([1, 1])
                    col1.markdown("Recommendations")
                    col1.write(inference_model.recommend(user=recommend_user, N=recommend_n, as_df=True))
                    col2.markdown("True ratings")
                    col2.write(inference_model.get_user_ratings(user=recommend_user, as_df=True))

                opts_it = [inference_model.dataset_train.to_raw_iid(x) for x in
                           inference_model.dataset_train.all_items()]
                st.subheader("Find the top N similar item to an item")
                col1, col2 = st.beta_columns([2, 1])
                similar_item = col1.selectbox("Item", options=opts_it,
                                              help="Select an item to find the most similar other items.")
                similar_n = col2.number_input("N", min_value=1, step=1, value=10, key="N_items2",
                                              help="The number of top similar items")

                if similar_item:
                    st.write(inference_model.similar_items(similar_item, similar_n, use_df=True))
