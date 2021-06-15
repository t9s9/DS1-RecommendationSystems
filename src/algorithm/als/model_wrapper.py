from implicit.als import AlternatingLeastSquares
from dataset import Dataset


class ModelWrapper:
    def __init__(self, dataset: Dataset, iterations=50, factors=100, alpha=1, regularization=0.01):
        self.model = AlternatingLeastSquares(iterations=iterations,
                                             factors=factors,
                                             regularization=regularization)
        self.alpha = alpha
        self.dataset = dataset
        self.is_fitted = False

    def fit(self):
        self.model.fit((self.dataset.item_user * self.alpha))
        self.is_fitted = True

    def similar_items(self, item, N=10, show=False):
        sitems = self.model.similar_items(self.dataset.get_item_id(item), N=N)
        if show:
            for i, (idx, dist) in enumerate(sitems):
                print("{0:<3}{1:<20}{2:.3f}".format(i + 1, self.dataset.get_item(idx), dist))
        return sitems

    def recommend(self, user, N=10, show=False):
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