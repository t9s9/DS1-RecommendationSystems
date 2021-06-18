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

    def fit_callback(self, iteration, time):
        self.current_iteration = iteration

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

    def similar_items(self, item: str, N=10, show=False):
        sitems = self.model.similar_items(self.dataset_train.get_item_id(item), N=N)
        if show:
            for i, (idx, dist) in enumerate(sitems):
                print("{0:<3}{1:<20}{2:.3f}".format(i + 1, self.dataset_train.get_item(idx), dist))
        return sitems

    def recommend(self, user: str, N=10, show=False):
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

        return rec

    def evaluate(self, metric="map", k=10):
        if self.dataset_test is None:
            raise ValueError("No test dataset specified.")
        if metric not in ['map', 'precision']:
            raise ValueError(f"Unknown metric {metric}.")

        score = ranking_metrics_at_k(model=self.model, K=k, train_user_items=self.dataset_train.user_item,
                                     test_user_items=self.dataset_test.user_item)[metric]
        return dict(metric=metric, k=k, score=score)

    def export(self):
        config = dict(iterations=self.iterations, factors=self.factors, regularization=self.regularization)
        data = dict(config=config)

        if self.is_fitted:
            data['user_factors'] = self.model.user_factors
            data['item_factors'] = self.model.item_factors
        return data
