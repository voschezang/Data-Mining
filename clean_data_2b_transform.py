import pickle
# from sklearn.ensemble import VotingRegressor
import pandas as pd
import numpy as np
from util import clustering
import gc
import util.data
from surprise import Reader, Dataset, SVD
import surprise.model_selection
seed = 123
np.random.seed(seed)

# TODO
# data_all = pd.read_csv(
# 'data/training_set_VU_DM_clean.csv', sep=';', nrows=5 * 1000)
data_all = pd.read_csv('data/training_set_VU_DM_clean.csv', sep=';')

n = 10
# n = 2  # TODO
n_chuncks = 10
k_user = 'AffinityPropagation'
k_item = 'FeatureAgglomeration'
k_user_long = clustering.USER_KEY_PREFIX + k_user
k_item_long = clustering.ITEM_KEY_PREFIX + k_item
keys_search, keys_property, _, _ = clustering.init(data_all)
models_user = []
models_item = []


model_user = clustering.load_user_model(n)
print('\nTransform training data (users)', model_user)
for indices in np.array_split(np.arange(data_all.shape[0]), n_chuncks):
    result = clustering.predict(data_all.loc[indices], {k_user: model_user},
                                keys_search, 'srch_id', clustering.USER_KEY_PREFIX)
    # users = users[clustering.USER_KEY_PREFIX + k_user]
    data_all.loc[indices, k_user_long] = result[k_user_long]
    result = None
    gc.collect()
# enforce disallocation of memory
clustering.gc_collect(model_user)

model_item = clustering.load_item_model(n)
print('Transform training data (items)')
for indices in np.array_split(np.arange(data_all.shape[0]), n_chuncks):
    result = clustering.predict(data_all.loc[indices], {k_item: model_item},
                                keys_property, 'prop_id', clustering.ITEM_KEY_PREFIX)
    # items = items[clustering.ITEM_KEY_PREFIX + k_item]
    data_all.loc[indices, k_item_long] = result[k_item_long]
    result = None
    gc.collect()
# enforce disallocation of memory
clustering.gc_collect(model_item)

print('\nsetup scores df (1)')
k_user = clustering.USER_KEY_PREFIX + k_user
k_item = clustering.ITEM_KEY_PREFIX + k_item

data_all_clusters = data_all[[k_user, k_item, 'score']]
data_all_clusters.to_csv(
    'data/clustering_data_train.csv', sep=';', index=False)
data_all = None
gc.collect()

print('setup scores df (2)')
scores_train = util.data.scores_df(data_all_clusters, k_user, k_item)
scores_train.to_csv('data/clustering_scores_train.csv',
                    sep=';', index=True)
scores_train_ = Dataset.load_from_df(scores_train, Reader(rating_scale=(0, 5)))

trainset, _ = surprise.model_selection.train_test_split(
    scores_train_, test_size=1e-15, random_state=seed)
print('fit SVD')
model = SVD()
model.fit(trainset)
with open('data/svd_model.pkl', 'wb') as f:
    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
