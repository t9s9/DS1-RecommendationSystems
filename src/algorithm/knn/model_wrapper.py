import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import ranking_metrics_at_k
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split


from src.frontend.dataset import DatasetWrapper
from surprise.prediction_algorithms.knns import KNNWithMeans
import requests


def is_item(item,all_items):
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


class KNNModelWrapper:
    def __init__(self, dataset: DatasetWrapper, iterations=50, k=20,test_size=0.1):
        self.iterations = iterations
        self.k = k
        self.test_size = test_size

        reader = Reader(rating_scale=(0, 1))
        self.dataset = dataset.data
        data = Dataset.load_from_df(dataset.data, reader)
        self.dataset_train, self.dataset_test = train_test_split(data,test_size=self.test_size)

        self.is_fitted = False

        self.model = KNNWithMeans(sim_options={"user_based": True, "name":"pearson"},k=self.k,min_k=5)
        self.model_item = KNNWithMeans(sim_options={"user_based": False, "name":"pearson"},k=self.k,min_k=5)
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

    def derive_from(self,train):
        self.model = train["user_factors"]
        self.model_item = train["item_factors"]
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
        sitems = self.model_item.get_neighbors(raw,N)
        sitems = [[self.dataset_train.to_raw_iid(x),0] for x in sitems]
        sitems.sort(key=lambda x: x[0], reverse=True)
        sitems = sitems[:N]
        if use_df:
            sitems = pd.DataFrame(sitems, columns=['item', 'score'])
        return sitems

    def recommend(self, user: str, N=10, show=False, as_df=False):
        list_items_filtered = []
        list_items_names = []
        ratings = []
        for x in self.dataset_train.all_items():
            raw = self.dataset_train.to_raw_iid(x)
            if is_item(raw,self.item_dict):
                pred = self.model.predict(user,self.dataset_train.to_raw_iid(x))
                ratings.append([raw,pred[3]])


        ratings.sort(key=lambda x: x[1], reverse=True)
        rec = ratings[:N]
        print(ratings)
        if as_df:
            rec = pd.DataFrame(rec, columns=['item', 'score'])
        return rec

    def get_user_ratings(self, user, as_df=False):
        userid = self.dataset[self.dataset.iloc[:,0] == user]
        rating = {"item": userid.iloc[:,1], "rating": userid.iloc[:,2]}
        if as_df:
            rating = pd.DataFrame(rating, columns=['item', 'rating'])
        return rating

    def evaluate(self, metric="map", k=10):
        if self.dataset_test is None:
            raise ValueError("No test dataset specified.")
        if metric not in ['map', 'precision']:
            raise ValueError(f"Unknown metric {metric}.")

        return dict(metric=metric, k=k, score=1.0)

    def export(self):
        config = dict()
        data = dict(config=config)

        if self.is_fitted:
            data['user_factors'] = self.model
            data['item_factors'] = self.model_item
        return data

    def as_inference(self, user_factors, item_factors):
        self.model.user_factors = user_factors
        self.model.item_factors = item_factors
        return self
