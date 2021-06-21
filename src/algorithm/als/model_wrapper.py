import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import ranking_metrics_at_k

from src.frontend.dataset import DatasetWrapper


class ImplicitModelWrapper:
    def __init__(self, dataset: DatasetWrapper, iterations=50, factors=100, regularization=0.01,
                 test_size=0.1):
        self.iterations = iterations
        self.factors = factors
        self.regularization = regularization
        self.test_size = test_size

        self.dataset_train, self.dataset_test = self.resolve_dataset(dataset)
        self.is_fitted = False

        self.model = AlternatingLeastSquares(iterations=self.iterations,
                                             factors=self.factors,
                                             regularization=self.regularization)

        self.model.fit_callback = self.fit_callback

        self.current_iteration = 0
        self.current_iteration_time = 0

    def fit_callback(self, iteration, time):
        self.current_iteration = iteration
        self.current_iteration_time = time

    def resolve_dataset(self, dataset: DatasetWrapper):
        """
        If the dataset is a DatasetWrapper it contains the information we can use for training.
        """
        if dataset.sparse_data_train is None:
            return dataset.create_sparse_dataset(test_size=self.test_size)
        else:
            return dataset.sparse_data_train, dataset.sparse_data_test

    def fit(self):
        self.model.fit(self.dataset_train.item_user)
        self.is_fitted = True
        return self

    def similar_items(self, item: str, N=10, show=False, use_df=False):
        sitems = self.model.similar_items(self.dataset_train.get_item_id(item), N=N+1)
        if show:
            for i, (idx, dist) in enumerate(sitems):
                print("{0:<3}{1:<20}{2:.3f}".format(i + 1, self.dataset_train.get_item(idx), dist))
        if use_df:
            sitems = pd.DataFrame(sitems, columns=['item', 'score'])
            sitems['item'] = sitems['item'].apply(lambda x: self.dataset_train.get_item(x))
            sitems = sitems.tail(-1)  # skip the first item because its itself
        return sitems

    def recommend(self, user: str, N=10, show=False, as_df=False):
        userid = self.dataset_train.get_user_id(user)

        rec = self.model.recommend(userid, self.dataset_train.user_item, N=N, filter_already_liked_items=True)
        if show:
            for i, (idx, dist) in enumerate(rec):
                print("{0:<3}{1:<20}{2:.3f}".format(i + 1, self.dataset_train.get_item(idx), dist))
            print("-" * 30)
            print("True feedback:")
            for subreddit, rating in sorted(zip(self.dataset_train.user_item.getrow(userid).indices,
                                                self.dataset_train.user_item.getrow(userid).data), key=lambda x: x[1],
                                            reverse=True):
                print("{0:<23}{1:<3}".format(self.dataset_train.get_item(subreddit), rating))
        if as_df:
            rec = pd.DataFrame(rec, columns=['item', 'score'])
            rec['item'] = rec['item'].apply(lambda x: self.dataset_train.get_item(x))

        return rec

    def get_user_ratings(self, user, as_df=False):
        userid = self.dataset_train.get_user_id(user)
        rating = sorted(zip(self.dataset_train.user_item.getrow(userid).indices,
                            self.dataset_train.user_item.getrow(userid).data), key=lambda x: x[1], reverse=True)
        if as_df:
            rating = pd.DataFrame(rating, columns=['item', 'rating'])
            rating['item'] = rating['item'].apply(lambda x: self.dataset_train.get_item(x))
        return rating

    def get_user_ratings_test(self, user, as_df=False):
        userid = self.dataset_train.get_user_id(user)
        rating = sorted(zip(self.dataset_test.user_item.getrow(userid).indices,
                            self.dataset_test.user_item.getrow(userid).data), key=lambda x: x[1], reverse=True)
        if as_df:
            rating = pd.DataFrame(rating, columns=['item', 'rating'])
            rating['item'] = rating['item'].apply(lambda x: self.dataset_train.get_item(x))
        return rating

    def evaluate(self, metric="map", k=10):
        if self.dataset_test is None:
            raise ValueError("No test dataset specified.")
        if metric not in ['map', 'precision']:
            raise ValueError(f"Unknown metric {metric}.")

        score = ranking_metrics_at_k(model=self.model, K=k, train_user_items=self.dataset_train.user_item,
                                     test_user_items=self.dataset_test.user_item)[metric]
        return dict(score=score, k=k, metric=metric)

    def export(self):
        config = dict(iterations=self.iterations, factors=self.factors, regularization=self.regularization)
        data = dict(config=config)

        if self.is_fitted:
            data['user_factors'] = self.model.user_factors
            data['item_factors'] = self.model.item_factors
        return data

    def as_inference(self, user_factors, item_factors):
        self.model.user_factors = user_factors
        self.model.item_factors = item_factors
        return self
