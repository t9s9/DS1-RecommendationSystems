from typing import Union

from implicit.als import AlternatingLeastSquares

from src.algorithm.als.sparsedataset import SparseDataset
from src.frontend.dataset import DatasetWrapper


class ImplicitModelWrapper:
    def __init__(self, dataset: Union[SparseDataset, DatasetWrapper], iterations=50, factors=100, regularization=0.01):
        self.iterations = iterations
        self.factors = factors
        self.regularization = regularization

        self.dataset = self.resolve_dataset(dataset)
        self.is_fitted = False

        self.model = AlternatingLeastSquares(iterations=self.iterations,
                                             factors=self.factors,
                                             regularization=self.regularization)

    def resolve_dataset(self, dataset: Union[SparseDataset, DatasetWrapper]):
        """
        If the dataset is a DatasetWrapper it contains the information we can use for training.
        """
        if isinstance(dataset, SparseDataset):
            return dataset
        elif isinstance(dataset, DatasetWrapper):
            if dataset.sparse_data is None:
                return dataset.create_sparse_dataset()

    def fit(self):
        self.model.fit(self.dataset.item_user)
        self.is_fitted = True
        return self

    def similar_items(self, item: str, N=10, show=False):
        sitems = self.model.similar_items(self.dataset.get_item_id(item), N=N)
        if show:
            for i, (idx, dist) in enumerate(sitems):
                print("{0:<3}{1:<20}{2:.3f}".format(i + 1, self.dataset.get_item(idx), dist))
        return sitems

    def recommend(self, user: str, N=10, show=False):
        userid = self.dataset.get_user_id(user)

        rec = self.model.recommend(userid, self.dataset.user_item, N=N, filter_already_liked_items=True)
        if show:
            for i, (idx, dist) in enumerate(rec):
                print("{0:<3}{1:<20}{2:.3f}".format(i + 1, self.dataset.get_item(idx), dist))
            print("-" * 30)
            print("True feedback:")
            for subreddit, rating in sorted(zip(self.dataset.user_item.getrow(userid).indices,
                                                self.dataset.user_item.getrow(userid).data), key=lambda x: x[1],
                                            reverse=True):
                print("{0:<23}{1:<3}".format(self.dataset.get_item(subreddit), rating))

        return rec

    def evaluate(self):
        pass
