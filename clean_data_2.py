import gc
import sklearn.cluster
import sklearn.metrics
import sklearn.neighbors
import sklearn.linear_model
import sklearn.tree
import sklearn.svm
import sklearn.ensemble
import sklearn
import pandas as pd
import numpy as np
seed = 123
np.random.seed(seed)


def extract_data(data, keys, k='srch_id'):
    # select unique rows, based on keys
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


def sample(data, keys, k='srch_id', rm_first_column=True):
    # sample & shuffle
    data_unique_rows = extract_data(data, keys, k)
    sample = data_unique_rows.sample(frac=1, random_state=seed)
    return sample


class FeatureAgglomeration(sklearn.cluster.FeatureAgglomeration):
    def predict(self, x_test):
        cluster_distances = self.transform(x_test)
        return np.argmin(cluster_distances, axis=1)


if __name__ == "__main__":
    data = pd.read_csv('data/training_set_VU_DM_clean.csv',
                       sep=';', nrows=5 * 1000)
    # TODO
    # data = pd.read_csv('data/training_set_VU_DM_clean.csv', sep=';')

    # data.drop(columns=['position'], inplace=True)
    # for k in data.columns:
    #     if data[k].isna().sum() > 0:
    #         #         print('rm %0.4f' % (data_all[k].isna().sum() / data_all.shape[0]), k)
    #         data.drop(columns=[k], inplace=True)

    # split columns into srch and prop
    # rm irrelevant info
    keys_property = [k for k in data.columns if ('comp' in k or 'prop' in k or 'site' in k or 'price_usd' in k)
                     and k not in ['prop_id']]
    # keys_other += ['click_bool', 'booking_bool', 'gross_bookings_usd', 'random_bool', 'score', 'price_usd', 'position']
    keys_search = [k for k in data.columns
                   if k not in keys_property
                   and k not in ['srch_id', 'travel_distance', 'travel_distances',
                             'click_bool', 'booking_bool', 'random_bool'] and
                   'orig' not in k
                   and 'cluster' not in k]

    n_clusters = 100
    models_user = {'KMeans': sklearn.cluster.KMeans(n_clusters, n_jobs=2, random_state=seed),
                   'FeatureAgglomeration': FeatureAgglomeration(n_clusters),
                   'AffinityPropagation': sklearn.cluster.AffinityPropagation(convergence_iter=15, damping=0.5, max_iter=200)
                   }

    models_item = {'KMeans': sklearn.cluster.KMeans(n_clusters, n_jobs=2, random_state=seed),
                   'FeatureAgglomeration': FeatureAgglomeration(n_clusters),
                   'AffinityPropagation': sklearn.cluster.AffinityPropagation(convergence_iter=15, damping=0.5, max_iter=200)
                   }
    # train user model
    x_train_users = sample(data, keys_search, k='srch_id')
    for k, model in models_user.items():
        model.fit(x_train_users)
        data['cluster_id_users_' + k] = model.predict(data[keys_search])
    # clear memory
    x_train_users = None
    gc.collect()

    # train item model
    x_train_items = sample(data, keys_property, k='prop_id')
    for k, model in models_item.items():
        model.fit(x_train_items)
        data['cluster_id_items_' + k] = model.predict(data[keys_property])
    # TODO
    # data.to_csv('data/training_set_VU_DM_clean.csv', sep=';', index=False)
    # clear memory
    x_train_items = None
    data = None
    gc.collect()

    # transform test data
    data = pd.read_csv('data/training_set_VU_DM_clean.csv',
                       sep=';', nrows=1000)
    # TODO
    # data = pd.read_csv('data/training_set_VU_DM_clean.csv', sep=';')
    # make predictions on test data
    for k, model in models_user.items():
        data['cluster_id_users_' + k] = model.predict(data[keys_search])

    for k, model in models_item.items():
        data['cluster_id_items_' + k] = model.predict(data[keys_property])

    # TODO
    # data.to_csv('data/test_set_VU_DM_clean.csv', sep=';', index=False)
