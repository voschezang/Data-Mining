import pickle
# from sklearn.ensemble import VotingRegressor
import pandas as pd
import numpy as np
from util import clustering
seed = 123
np.random.seed(seed)

# TODO
# data_all = pd.read_csv(
# 'data/training_set_VU_DM_clean.csv', sep=';', nrows=5 * 1000)
data_all = pd.read_csv('data/training_set_VU_DM_clean.csv', sep=';')

data_all_clusters = pd.read_csv('data/clustering_data_train.csv', sep=';')

n = 10
# n = 2  # TODO
k_user = 'AffinityPropagation'
k_item = 'FeatureAgglomeration'
# k_user_long = clustering.USER_KEY_PREFIX + k_user
# k_item_long = clustering.ITEM_KEY_PREFIX + k_item
keys_search, keys_property, _, _ = clustering.init(data_all)
k_user = clustering.USER_KEY_PREFIX + k_user
k_item = clustering.ITEM_KEY_PREFIX + k_item

with open('data/svd_model.pkl', 'rb') as f:
    model = pickle.load(f)

scores_train = pd.read_csv(
    'data/clustering_scores_train.csv', sep=';', index_col=0)

# fill in predicted training data
print('predict training data with SVD')
score_svd = clustering.svd_predict(model, scores_train, data_all)
data_all['score_svd'] = score_svd
# assert 'score_svd' in data_all.columns
# assert not score_svd.isna().any()
# data_all['score_svd'] = pd.Series()
# scores_pred = clustering.svd_predict(model, scores_train)
# scores_pred
# print('cp scores pred to df')
# for i, row in data_all.iterrows():
#     #     score_pred = scores_pred[row[k_user]][row[k_item]]
#     data_all['score_svd'] = scores_pred[row[k_user]][row[k_item]]

for k in data_all.columns:
    if 'cluster' in k:
        data_all.drop(columns=[k], inplace=True)

print('save training data to disk')
data_all.to_csv('data/training_set_VU_DM_clean_2.csv',
                sep=';', index=False)
# data_all = None
# clear memory
# gc.collect()
print('\ndone')
