import gc
import pickle
# from sklearn.ensemble import VotingRegressor
import pandas as pd
import numpy as np
from util import clustering
import util.data
seed = 123
np.random.seed(seed)

# predict test data
print('\nPredict test data')
# read training data for init
data_all = pd.read_csv(
    'data/training_set_VU_DM_clean.csv', sep=';', nrows=10000)
# data = pd.read_csv('data/test_set_VU_DM_clean.csv', sep=';', nrows=1 * 1000)
data = pd.read_csv('data/test_set_VU_DM_clean.csv', sep=';')
util.data.rm_na(data)

n = 10
# n = 2  # TODO
k_user = 'AffinityPropagation'
k_item = 'FeatureAgglomeration'
# k_user_long = clustering.USER_KEY_PREFIX + k_user
# k_item_long = clustering.ITEM_KEY_PREFIX + k_item
keys_search, keys_property, _, _ = clustering.init(data_all)
k_user = clustering.USER_KEY_PREFIX + k_user
k_item = clustering.ITEM_KEY_PREFIX + k_item
n_chuncks = 100

data_clusters = data[['srch_id', 'prop_id']].copy()
data_clusters['user'] = pd.Series()
data_clusters['item'] = pd.Series()

model_user = clustering.load_user_model(n)
for indices in np.array_split(np.arange(data.shape[0]), n_chuncks):
    print('\t pred user chunck')
    users = model_user.predict(data.loc[indices, keys_search])
    data_clusters.loc[indices, 'user'] = users
    users = None
    gc.collect()
clustering.gc_collect(model_user)
data_clusters.to_csv(
    'data/clustering_data_test_1.csv', sep=';', index=False)

model_item = clustering.load_item_model(n)
for indices in np.array_split(np.arange(data.shape[0]), n_chuncks):
    print('\t pred item chunck')
    items = model_item.predict(data.loc[indices, keys_property])
    data_clusters.loc[indices, 'items'] = items
    items = None
    gc.collect()
clustering.gc_collect(model_item)

data_clusters.to_csv(
    'data/clustering_data_test.csv', sep=';', index=False)


print('\nDone')
