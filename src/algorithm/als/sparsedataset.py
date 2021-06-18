import pandas as pd
from implicit.evaluation import train_test_split
from scipy.sparse import csr_matrix, coo_matrix


class SparseDataset:
    @classmethod
    def from_csv(cls, csv, user=None, item=None, rating=None, test_size=0.0):
        print("Loading csv.")
        return cls.from_dataframe(pd.read_csv(csv), user=user, item=item, rating=rating, test_size=test_size)

    @classmethod
    def from_dataframe(cls, df, user=None, item=None, rating=None, test_size=0.0):
        user = df.columns[0] if user is None else user
        item = df.columns[1] if item is None else item
        rating = df.columns[2] if rating is None else rating
        print("Creating pivot.")
        pivot = pd.pivot_table(df, index=item, columns=user, values=rating, fill_value=0)
        return cls.from_pivot(pivot, test_size=test_size)

    @classmethod
    def from_pivot(cls, pivot, test_size=0.0):
        if test_size == 0.0:
            return cls(pivot.values, items=pivot.index, users=pivot.columns), None
        elif 0 < test_size <= 1.0:
            coo = coo_matrix(pivot.values, dtype=float)

            train, test = train_test_split(coo, train_percentage=1 - test_size)
            # init train with all metadata
            return cls(train, items=pivot.index, users=pivot.columns), cls(test)
        else:
            raise ValueError("Test size needs to be between 0.0 and 1.0.")

    def __init__(self, item_user_matrix, items=None, users=None):
        self.item_user = csr_matrix(item_user_matrix)
        self.user_item = self.item_user.T.tocsr()

        self.items = items
        self.users = users

        if items is not None:
            self.item_to_idx = {item: idx for idx, item in enumerate(items)}
            self.idx_to_item = {idx: item for idx, item in enumerate(items)}
        if users is not None:
            self.user_to_idx = {item: idx for idx, item in enumerate(users)}
            self.idx_to_user = {idx: item for idx, item in enumerate(users)}

    def num_items(self):
        return self.item_user.shape[0]

    def num_users(self):
        return self.item_user.shape[1]

    def get_user(self, idx):
        return self.idx_to_user[idx]

    def get_user_id(self, user_name):
        return self.user_to_idx[user_name]

    def get_item(self, idx):
        return self.idx_to_item[idx]

    def get_item_id(self, item_name):
        return self.item_to_idx[item_name]
