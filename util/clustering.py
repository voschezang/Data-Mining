import collections
import gc
import pickle
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.cluster
import sklearn.metrics
import pandas as pd
import sklearn.neighbors
import sklearn.linear_model
import sklearn.tree
import sklearn.svm
seed = 123
np.random.seed(seed)

USER_KEY_PREFIX = 'cluster_id_users_'
ITEM_KEY_PREFIX = 'cluster_id_items_'


class FeatureAgglomeration(sklearn.cluster.FeatureAgglomeration):
    def predict(self, x_test):
        cluster_distances = self.transform(x_test)
        return np.argmin(cluster_distances, axis=1)


class VotingRegressor:
    # TODO update sklearn and use sklearn.ensemble.VotingRegressor
    def __init__(self, estimators):
        # estimators = list of tuples (key, estimator)
        print('init VotingRegressor')
        self.estimators = estimators

    def predict(self, X):
        predictions_per_est = []
        print('\tpredict X', self)
        assert self.estimators is not None
        for i, (_, est) in enumerate(self.estimators):
            predictions_per_est .append(est.predict(X))
        return np.median(predictions_per_est, axis=0)


def init(data, n_clusters=50):
    assert data.shape[0] > 10 * n_clusters
    # split columns into srch and prop
    # rm irrelevant info
    keys_property = [k for k in data.columns if ('comp' in k or 'prop' in k or 'site' in k or 'price_usd' in k) and
                     k not in ['prop_id']]
    # keys_other += ['click_bool', 'booking_bool', 'gross_bookings_usd', 'random_bool', 'score', 'price_usd', 'position']
    keys_search = [k for k in data.columns
                   if k not in keys_property
                   and k not in ['srch_id', 'score', 'position', 'travel_distance',
                             'travel_distances', 'click_bool',
                                 'booking_bool', 'random_bool'] and
                   'orig' not in k and
                   'cluster' not in k]

    assert [k in data.columns for k in keys_property]
    assert [k in data.columns for k in keys_search]
    # TODO use MiniBatchKMeans
    models_user = {'KMeans': sklearn.cluster.KMeans(n_clusters, n_jobs=2, random_state=seed),
                   'FeatureAgglomeration':
                   FeatureAgglomeration(n_clusters=24),
                   'AffinityPropagation': sklearn.cluster.AffinityPropagation(convergence_iter=15, damping=0.5, max_iter=30)
                   }

    models_item = {'KMeans': sklearn.cluster.KMeans(n_clusters, n_jobs=2, random_state=seed),
                   'FeatureAgglomeration': FeatureAgglomeration(n_clusters=24),
                   'AffinityPropagation': sklearn.cluster.AffinityPropagation(convergence_iter=15, damping=0.5, max_iter=50)
                   }
    init_df_columns(data, models_user, models_item)
    return keys_search, keys_property, models_user, models_item


def init_df_columns(data, models_user, models_item):
    for k in models_user.keys():
        data[USER_KEY_PREFIX + k] = pd.Series()
    for k in models_item.keys():
        data[ITEM_KEY_PREFIX + k] = pd.Series()


def fit(data, models, keys, k, shuffle=False):
    x_train = sample_and_shuffle(data, keys, k=k, shuffle=shuffle)
    for model in models.values():
        print('fit model', model, x_train.shape)
        model.fit(x_train)


def predict(data, models, keys, k, prefix):
    # return predictions of each model
    # TODO rm k in farg
    indices = data.index
    prediction_keys = []
    data_pred = data[[k]].copy()
    for k, model in models.items():
        print('\tPredict using %s (k: `%s`)' % (k, prefix + k), model)
        assert model.estimators is not None
        y_pred = model.predict(data[keys])
        key = prefix + k
        print('save pred to data')
        data_pred[key] = pd.Series()
        data_pred.loc[indices, key] = y_pred
        prediction_keys.append(key)
    return data_pred[prediction_keys]


def extract_data(data, keys, k='srch_id'):
    # select unique rows, based on keys
    print('\textract_data(k: %s)' % k)
    return data[keys + [k]].drop_duplicates(subset=k)[keys]


def sample_and_shuffle(data, keys, k='srch_id', rm_first_column=True, shuffle=False):
    # sample & shuffle
    data_unique_rows = extract_data(data, keys, k)
    print('n_keys', len(keys + [k]), k)
    if shuffle:
        return data_unique_rows.sample(frac=1, random_state=seed)
    return data_unique_rows


def svd_predict(model, scores: pd.DataFrame, data_all_clusters=None, k_user='user', k_item='item'):
    # data_all_clusters :: {index, k_user, k_item, true score}
    # Return a df {user: {item: predicted score}}
    if data_all_clusters is not None:
        scores = None
        data = data_all_clusters[['score']].copy()
        # make individual predicitons, to reduce memory overhead
        # data_all['score_svd'] = pd.Series()
        # for i, row in scores.iterrows():
        for i, row in data_all_clusters.iterrows():
            i_n = 100 * 1000
            if i % i_n == 0:
                print('\tsvd predict, row: %i \t* %i \t /%i' %
                      (i / i_n, i_n, data_all_clusters.shape[0]))
            # for i, row in data_all.iterrows():
            # row = scores.loc[i]
            item = row[k_item]
            user = row[k_user]
            result = model.predict(str(item), str(user), verbose=0)
            data.loc[i, 'score_svd'] = result.est
        return data['score_svd']

    results = collections.defaultdict(dict)
    # results = pd.DataFrame()
    for _, row in scores.iterrows():
        item = row['item']
        user = row['user']
        result = model.predict(str(item), str(user), verbose=0)
        # results[user][item] = result.est
        results[user][item] = result.est
    return results


def load_user_model(n):
    models_user = []
    for i in range(n):
        print(i, 'data/est_user_%i.pkl' % i)
        with open('data/est_user_%i.pkl' % i, 'rb') as f:
            models_user.append(pickle.load(f))

    model_user = VotingRegressor(
        [(str(i), reg) for i, reg in enumerate(models_user)])
    return model_user


def load_item_model(n):
    models_item = []
    for i in range(n):
        print(i, 'data/est_item_%i.pkl' % i)
        with open('data/est_item_%i.pkl' % i, 'rb') as f:
            models_item.append(pickle.load(f))

    model_item = VotingRegressor(
        [(str(i), reg) for i, reg in enumerate(models_item)])
    return model_item


def gc_collect(ensemble_model: VotingRegressor):
    for i, _ in enumerate(ensemble_model.estimators):
        ensemble_model.estimators[i] = None
    gc.collect()
