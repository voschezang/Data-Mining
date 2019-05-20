import collections
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


def init(data, n_clusters=10):
    assert data.shape[0] > 10 * n_clusters
    # split columns into srch and prop
    # rm irrelevant info
    keys_property = [k for k in data.columns if ('comp' in k or 'prop' in k or 'site' in k or 'price_usd' in k)
                     and k not in ['prop_id']]
    # keys_other += ['click_bool', 'booking_bool', 'gross_bookings_usd', 'random_bool', 'score', 'price_usd', 'position']
    keys_search = [k for k in data.columns
                   if k not in keys_property and
                   k not in ['srch_id', 'score', 'position', 'travel_distance',
                                 'travel_distances', 'click_bool',
                                 'booking_bool', 'random_bool'] and
                   'orig' not in k
                   and 'cluster' not in k]

    assert [k in data.columns for k in keys_property]
    assert [k in data.columns for k in keys_search]
    models_user = {'KMeans': sklearn.cluster.KMeans(n_clusters, n_jobs=2, random_state=seed),
                   'FeatureAgglomeration': FeatureAgglomeration(n_clusters),
                   # 'AffinityPropagation': sklearn.cluster.AffinityPropagation(convergence_iter=15, damping=0.5, max_iter=50)
                   }

    models_item = {'KMeans': sklearn.cluster.KMeans(n_clusters, n_jobs=2, random_state=seed),
                   'FeatureAgglomeration': FeatureAgglomeration(n_clusters),
                   # 'AffinityPropagation': sklearn.cluster.AffinityPropagation(convergence_iter=15, damping=0.5, max_iter=50)
                   }
    return keys_search, keys_property, models_user, models_item


def init_df_columns(data, models_user, models_item):
    for k in models_user.keys():
        data[USER_KEY_PREFIX + k] = pd.Series()
    for k in models_item.keys():
        data[ITEM_KEY_PREFIX + k] = pd.Series()


def fit(data, models, keys, k):
    x_train = sample_and_shuffle(data, keys, k=k)
    for model in models.values():
        model.fit(x_train)


def predict(data, models, keys, k, prefix):
    indices = data.index
    prediction_keys = []
    for k, model in models.items():
        print('\t%s (k: `%s`)' % (k, prefix + k))
        y_pred = model.predict(data[keys])
        key = prefix + k
        data.loc[indices, key] = y_pred
        prediction_keys.append(key)
    return data[prediction_keys]

    # print('predict items')
    # for k, model in models_item.items():
    #     print('\t%s' % k)
    #     # data.loc[:, ITEM_KEY_PREFIX + k] = model.predict(data[keys_property])
    #     y_pred = model.predict(data[keys_property])
    #     data.loc[indices, key] = y_pred
    #     # for i in indices:
    #     # data.loc[indices[i], ITEM_KEY_PREFIX + k] = y_pred[i]


def extract_data(data, keys, k='srch_id'):
    # select unique rows, based on keys
    print('\textract_data(k: %s)' % k)
    data = data[keys + [k]]
    data_unique_rows = data.drop_duplicates(subset=k)
    assert data_unique_rows.shape[0] == data[k].unique().size, \
        'if key (srch_id) is equal, all non-property attributes should be equal as well'

    assert data_unique_rows[k].min(
    ) > 0, 'search id must be positive and nonzero'
    assert ~data_unique_rows[k].isna().any()
    assert ~data_unique_rows.isna().any().any()
    assert ~data_unique_rows.isin([np.nan, np.inf, -np.inf]).any().any()
    return data_unique_rows[keys]


def sample_and_shuffle(data, keys, k='srch_id', rm_first_column=True):
    # sample & shuffle
    data_unique_rows = extract_data(data, keys, k)
    sample = data_unique_rows.sample(frac=1, random_state=seed)
    return sample


def svd_predict(model, scores: pd.DataFrame):
    # Return a dict {user: {item: predicted score}}
    results = collections.defaultdict(dict)
    for _, row in scores.iterrows():
        item = row['item']
        user = row['user']
        result = model.predict(str(item), str(user), verbose=0)
        results[user][item] = result.est
    return results


class FeatureAgglomeration(sklearn.cluster.FeatureAgglomeration):
    def predict(self, x_test):
        cluster_distances = self.transform(x_test)
        return np.argmin(cluster_distances, axis=1)
