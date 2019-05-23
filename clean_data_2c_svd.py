import pickle
# from sklearn.ensemble import VotingRegressor
import pandas as pd
import numpy as np
from util import clustering
seed = 123
np.random.seed(seed)

# TODO
n_rows = 10000
# data_all = pd.read_csv(
#     'data/training_set_VU_DM_clean.csv', sep=';', nrows=n_rows)
# data_all_clusters = pd.read_csv(
#     'data/clustering_data_train.csv', sep=';',  nrows=n_rows)

# data_all = pd.read_csv('data/training_set_VU_DM_clean.csv', sep=';')
data_all = pd.read_csv('data/training_set_VU_DM_clean_2.csv', sep=';')
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

# with open('data/svd_model.pkl', 'rb') as f:
#     model = pickle.load(f)

scores_train = pd.read_csv(
    'data/clustering_scores_train.csv', sep=';', index_col=0)

# # fill in predicted training data
# print('predict training data with SVD')
# # score_svd :: [svd.predict(cluster.item, cluster.user)]
# cluster_score_svd = clustering.svd_predict(model, scores_train)
# with open('data/cluster_score_svd_dict_train.pkl', 'wb') as f:
#     pickle.dump(cluster_score_svd, f, pickle.HIGHEST_PROTOCOL)

with open('data/cluster_score_svd_dict_train.pkl', 'rb') as f:
    cluster_score_svd = pickle.load(f)

print('cp score svd')
data_all['score_svd'] = pd.Series()
for i, row in data_all_clusters.iterrows():
    i_n = 50 * 1000
    if i % i_n == 0:
        print('\tsvd predict, row: %i \t* %i \t /%i \t (%0.2f)' %
              (i / i_n, i_n, data_all_clusters.shape[0], i / data_all_clusters.shape[0]))

    user = row[k_user]
    item = row[k_item]
    data_all.loc[i, 'score_svd'] = cluster_score_svd[user][item]

    i_n = 500 * 1000
    if i % i_n == 0:
        print('\t saving')
        data_all.to_csv('data/training_set_VU_DM_clean_2.csv',
                        sep=';', index=False)
        print('\t\t saved')

print('\talmost done')

# score_svd = clustering.svd_predict(
#     model, None, data_all_clusters, k_user, k_item)
# data_all['score_svd'] = score_svd

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
