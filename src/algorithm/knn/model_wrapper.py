from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import KFold
from surprise.model_selection.validation import cross_validate
from surprise.prediction_algorithms.knns import KNNWithMeans

from src.frontend.dataset import DatasetWrapper


def is_item(item, all_items):
    try:
        if item == 0:
            return False
        if "into" in all_items[str(item)]:
            return all_items[str(item)]["into"] == []
        if all_items[str(item)]["gold"]["purchasable"] == True:
            return all_items[str(item)]["gold"]["total"] != 0
    except:
        return True
    return False


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


class KNNModelWrapper:
    def __init__(self, dataset: DatasetWrapper, k=20, sim="pearson"):
        self.k = k

        reader = Reader(rating_scale=(0, 1))
        self.dataset = dataset.data
        self.data = Dataset.load_from_df(dataset.data, reader)
        self.dataset_train = self.data.build_full_trainset()
        self.dataset_test = []

        self.is_fitted = False
        self.sim = sim

        self.model = KNNWithMeans(sim_options={"user_based": True, "name": sim}, k=self.k, min_k=5)
        self.model_item = KNNWithMeans(sim_options={"user_based": False, "name": sim}, k=self.k, min_k=5)
        self.model.fit_callback = self.fit_callback
        all_items = requests.get('http://ddragon.leagueoflegends.com/cdn/11.11.1/data/en_US/item.json').json()["data"]
        item_dict = {}
        all_item_names = []
        for i in all_items:
            all_items[i]["base_id"] = i
            item_dict[i] = all_items[i]
            item_dict[all_items[i]["name"]] = all_items[i]
            all_item_names.append(all_items[i]["name"])
        self.item_dict = item_dict

        self.current_iteration = 0
        self.current_iteration_time = 0

    def derive_from(self, train):
        self.model = train["user_factors"]
        self.model_item = train["item_factors"]
        self.dataset_test = train["testset"]
        self.dataset_train = train["trainset"]
        return self

    def fit_callback(self, iteration, time):
        self.current_iteration = iteration
        self.current_iteration_time = time

    def fit(self):
        self.model.fit(self.dataset_train)
        self.model_item.fit(self.dataset_train)
        self.is_fitted = True
        return self

    def similar_items(self, item: str, N=10, show=False, use_df=False):
        raw = self.dataset_train.to_inner_iid(item)
        sitems = self.model_item.get_neighbors(raw, N)
        sitems = [[self.dataset_train.to_raw_iid(x)] for x in sitems]
        sitems = sitems[:N]
        if use_df:
            sitems = pd.DataFrame(sitems, columns=['item'])
        return sitems

    def recommend(self, user: str, N=10, show=False, as_df=False):
        list_items_filtered = []
        list_items_names = []
        ratings = []
        for x in self.dataset_train.all_items():
            raw = self.dataset_train.to_raw_iid(x)
            if is_item(raw, self.item_dict):
                pred = self.model.predict(user, self.dataset_train.to_raw_iid(x))
                ratings.append([raw, pred[3]])
        self.dataset_train
        ratings.sort(key=lambda x: x[1], reverse=True)
        rec = ratings[:N]
        print(ratings)
        if as_df:
            rec = pd.DataFrame(rec, columns=['item', 'score'])
        return rec

    def get_user_ratings(self, user, as_df=False):
        userid = self.dataset[self.dataset.iloc[:, 0] == user]
        rating = {"item": userid.iloc[:, 1], "rating": userid.iloc[:, 2]}
        if as_df:
            rating = pd.DataFrame(rating, columns=['item', 'rating'])
        return rating

    def evaluate(self, metric="mae", k=10, cross_validation_folds=5, thresh=0.5):
        if self.dataset_test is None:
            raise ValueError("No test dataset specified.")
        if metric not in ['mse', 'mae', "map"]:
            raise ValueError(f"Unknown metric {metric}.")
        if metric == "map":
            kf = KFold(n_splits=5)
            results = dict(test_map=[], test_mar=[])
            for trainset, testset in kf.split(self.data):
                self.model.fit(trainset)
                predictions = self.model.test(testset)
                precisions, recalls = precision_recall_at_k(predictions, k=k, threshold=thresh)
                total = 0
                for x in precisions.keys():
                    total += precisions[x]
                results["test_map"].append(total / len(precisions))
                total = 0
                for x in recalls.keys():
                    total += recalls[x]
                results["test_mar"].append(total / len(recalls))
        else:
            results = cross_validate(self.model, self.data, measures=[metric], cv=cross_validation_folds,
                                     return_train_measures=True)

        return dict(score=np.mean(results["test_" + metric]), metric_k=k, sim_metric=self.sim, metric=metric, k=self.k,
                    cv_folds=cross_validation_folds)

    def export(self):
        config = dict()
        data = dict(config=config)

        if self.is_fitted:
            data['user_factors'] = self.model
            data['item_factors'] = self.model_item

        data["testset"] = self.dataset_test
        data["trainset"] = self.dataset_train
        return data

    def as_inference(self, user_factors, item_factors):
        self.model.user_factors = user_factors
        self.model.item_factors = item_factors
        return self
