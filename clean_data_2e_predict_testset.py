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


print('\n\n SVD predict test data')
data_clusters = pd.read_csv('data/clustering_data_test.csv', sep=';')

with open('data/svd_model.pkl', 'rb') as f:
    model = pickle.load(f)


data['score_svd'] = pd.Series()
for i, row in data.iterrows():
    i_n = 10 * 1000
    if i % i_n == 0:
        print('\tsvd predict test, row: %i \t*%i \t /%i' %
              (i / i_n, i_n, data.shape[0]))
    result = model.predict(str(data_clusters.at[i, 'item']), str(
        data_clusters.at[i, 'user']), verbose=0)
    data.loc[i, 'score_svd'] = result.est

for k in data.columns:
    if 'cluster' in k:
        data.drop(columns=[k], inplace=True)

print('save test data to disk')
data.to_csv('data/test_set_VU_DM_clean_2.csv', sep=';', index=False)
print('\nDone')
