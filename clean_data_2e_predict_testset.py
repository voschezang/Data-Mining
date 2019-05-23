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

# n_rows = 10 * 1000
# data = pd.read_csv('data/test_set_VU_DM_clean.csv', sep=';', nrows=n_rows)
# data_clusters = pd.read_csv(
#     'data/clustering_data_test.csv', sep=';', nrows=n_rows)

data = pd.read_csv('data/test_set_VU_DM_clean.csv', sep=';')
data_clusters = pd.read_csv('data/clustering_data_test.csv', sep=';')
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
# n_chuncks = 100


with open('data/svd_model.pkl', 'rb') as f:
    model = pickle.load(f)

print('SVD predict test data')

# print('setup scores df')
# # add dummy attr
# data_clusters['score'] = pd.Series()
# scores_test = util.data.scores_df(
#     data_clusters, 'user', 'items')  # `items` should be item
# scores_test.to_csv('data/clustering_scores_test.csv', sep=';', index=True)
scores_test = pd.read_csv('data/clustering_scores_test.csv', sep=';')


# print('predict test data with SVD')
# data['score'] = pd.Series()
# # score_svd = clustering.svd_predict(model, scores_test, data)
# score_svd = clustering.svd_predict(
#     model, None, data_clusters, 'user', 'items')
# data['score_svd'] = score_svd
# print('\t almost done')

# fill in predicted training data
print('predict test data with SVD')
# score_svd :: [svd.predict(cluster.item, cluster.user)]
cluster_score_svd = clustering.svd_predict(model, scores_test)
# scores_train['score_svd'] = cluster_score_svd
# scores_train.to_csv('data/clustering_scores_train_svd.csv',
# sep=';', index=False)
with open('data/cluster_score_svd_dict_test.pkl', 'wb') as f:
    pickle.dump(cluster_score_svd, f, pickle.HIGHEST_PROTOCOL)
# with open('data/cluster_score_svd_dict_test.pkl', 'rb') as f:
#     cluster_score_svd = pickle.load(f)

print('cp score svd')
data_all['score_svd'] = pd.Series()
for i, row in data_clusters.iterrows():
    i_n = 100 * 1000
    if i % i_n == 0:
        print('\tsvd predict, row: %i \t* %i \t /%i' %
              (i / i_n, i_n, data_clusters.shape[0]))
    user = row['user']
    item = row['items']
    data_all.loc[i, 'score_svd'] = cluster_score_svd[user][item]

    i_n = 500 * 1000
    if i % i_n == 0:
        print('\t saving')
        data_all.to_csv('data/test_set_VU_DM_clean_2.csv',
                        sep=';', index=False)
        print('\t\t saved')

print('\talmost done')
#
#
#
#
#
#
#
#

#
# data['score_svd'] = pd.Series()
# for i, row in data.iterrows():
#     i_n = 10 * 1000
#     if i % i_n == 0:
#         print('\tsvd predict test, row: %i \t*%i \t /%i' %
#               (i / i_n, i_n, data.shape[0]))
#     result = model.predict(str(data_clusters.at[i, 'item']), str(
#         data_clusters.at[i, 'user']), verbose=0)
#     data.loc[i, 'score_svd'] = result.est

for k in data.columns:
    if 'cluster' in k:
        data.drop(columns=[k], inplace=True)

print('save test data to disk')
# data.to_csv('data/test_set_VU_DM_clean_2.csv', sep=';', index=False)
print('\nDone')
