from pprint import pprint

import numpy as np
from implicit.evaluation import ranking_metrics_at_k
from scipy.sparse import csr_matrix
from sklearn.model_selection import ParameterGrid


def get_random_generator(state):
    if state is None or state is np.random:
        return np.random.mtrand._rand
    else:
        return np.random.RandomState(state)


class KFold:
    """
    Cross validation generator for sparse matrices

    Each fold is used once as a test set while the k - 1 remaining folds are
    used for training.
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        coo = X.tocoo()
        idx = np.arange(len(coo.data))

        if self.shuffle:
            get_random_generator(self.random_state).shuffle(idx)

        start, stop = 0, 0
        for fold_i in range(self.n_splits):
            start = stop
            stop += len(idx) // self.n_splits
            if fold_i < len(idx) % self.n_splits:
                stop += 1

            test_idx = idx[start:stop]
            train_idx = np.concatenate([idx[:start], idx[stop:]])

            train = csr_matrix((coo.data[train_idx],
                                (coo.row[train_idx], coo.col[train_idx])),
                               shape=coo.shape, dtype=coo.dtype)

            test = csr_matrix((coo.data[test_idx],
                               (coo.row[test_idx], coo.col[test_idx])),
                              shape=coo.shape, dtype=coo.dtype)
            yield train, test


class GridSearchCV:
    """
    Cross validated grid search on parameter grid for implicit algorithm.

    """

    def __init__(self, algo, param_grid, cv=5, eval_k=10, metrics=['map', 'precision']):
        self.algo = algo
        self.cv = cv
        self.param_grid = ParameterGrid(param_grid)
        self.eval_k = eval_k
        self.metrics = metrics
        self.result = dict()
        self.best = None

        self.cv = KFold(n_splits=cv, shuffle=True, random_state=0)

    def __len__(self):
        return len(self.param_grid)

    def fit(self, data):
        self.best = {metric: [0, "", ""] for metric in self.metrics}
        for param_set, params in enumerate(self.param_grid):
            self.result[f'param_set_{param_set}'] = dict(params=params)

            for fold, (train_data, test_data) in enumerate(self.cv.split(data)):
                if 'alpha' in params:
                    alpha = params['alpha']
                    del params['alpha']
                else:
                    alpha = 1
                algo = self.algo(**params)

                result = self._fit_and_eval(train_data=(train_data * alpha), test_data=(test_data * alpha), algo=algo)
                self.result[f'param_set_{param_set}'][f'Fold {fold}'] = result
            self.result[f'param_set_{param_set}']['mean'] = {key: np.mean(np.array(
                [self.result[f'param_set_{param_set}'][f'Fold {fold}'][key] for fold in range(self.cv.n_splits)])) for
                key in self.metrics}
            self.result[f'param_set_{param_set}']['std'] = {key: np.std(np.array(
                [self.result[f'param_set_{param_set}'][f'Fold {fold}'][key] for fold in range(self.cv.n_splits)])) for
                key in self.metrics}
            for metric in self.metrics:
                if self.best[metric][0] < self.result[f'param_set_{param_set}']['mean'][metric]:
                    self.best[metric] = [self.result[f'param_set_{param_set}']['mean'][metric],
                                         self.result[f'param_set_{param_set}']['params'],
                                         f'param_set_{param_set}']
        return self

    def _fit_and_eval(self, train_data, test_data, algo):
        algo.fit(train_data)
        eval = ranking_metrics_at_k(model=algo,
                                    train_user_items=train_data.T.tocsr(),
                                    test_user_items=test_data.T.tocsr(),
                                    K=self.eval_k,
                                    show_progress=False)
        return {key: eval[key] for key in self.metrics}

    def get_result(self, show=True):
        if show:
            pprint(self.result)
        return self.result

    def get_best(self):
        return self.best
