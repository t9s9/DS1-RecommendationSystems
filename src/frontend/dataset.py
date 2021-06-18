from src.algorithm.als.sparsedataset import SparseDataset
from surprise.dataset import Dataset, Reader


class DatasetWrapper:
    def __init__(self, name: str, id: int, data=None, param=None):
        self.id = id
        self.parameter = param
        self.name = name

        self.data = data
        self.sparse_data_train, self.sparse_data_test = None, None
        self.surprise_data = None

        self.trainings = []

    def create_sparse_dataset(self, test_size=0.0):
        self.sparse_data_train, self.sparse_data_test = SparseDataset.from_dataframe(self.data, test_size=test_size)
        return self.sparse_data_train, self.sparse_data_test

    def create_surprise_dataset(self):
        reader = Reader(rating_scale=(1, 5))
        self.surprise_data = Dataset.load_from_df(self.data, reader)
        return self.surprise_data

    def __str__(self):
        return f"{self.name} data_shape: {self.data.shape} has_sparse_train: {self.sparse_data_train is not None}" \
               f" has_sparse_test: {self.sparse_data_test is not None} id: {id(self)}"
