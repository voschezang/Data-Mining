import pickle
import sklearn.cluster
import sklearn
# from sklearn.ensemble import VotingRegressor
import pandas as pd
import numpy as np
from util import clustering
import util.data
seed = 123
np.random.seed(seed)


# data_all = pd.read_csv(
# 'data/training_set_VU_DM_clean.csv', sep=';', nrows=10 * 1000)
data_all = pd.read_csv('data/training_set_VU_DM_clean.csv', sep=';')

n = 10
folds = util.data.cv_folds_for_sklearn(
    data_all, n_cv_folds=n, resampling_ratio=0)
i_majority, i_minority = folds[0]
print('len i_train: %i' % i_minority.size)

keys_search, keys_property, _, _ = clustering.init(
    data_all)

k_user = 'AffinityPropagation'
k_item = 'FeatureAgglomeration'
# use ensemble of cluster-algos, each trained on a different trainingset
# use equal weights (assume enough variance)
models_user = [
    sklearn.cluster.AffinityPropagation(
        convergence_iter=15, damping=0.5, max_iter=30)
    for _ in folds]
models_item = [
    clustering.FeatureAgglomeration(n_clusters=24)
    for _ in folds]

# TODO save models to disk

# fit
print('Fit training data')
# use the minority indices (i_test)
for i, (_, i_train) in enumerate(folds):
    clustering.fit(data_all.loc[i_train], {
                   k_user: models_user[i]}, keys_search, 'srch_id')
    clustering.fit(data_all.loc[i_train], {
                   k_item: models_item[i]}, keys_property, 'prop_id')

for i, est in enumerate(models_user):
    with open('data/est_user_%i.pkl' % i, 'wb') as f:
        pickle.dump(est, f, pickle.HIGHEST_PROTOCOL)

for i, est in enumerate(models_item):
    with open('data/est_item_%i.pkl' % i, 'wb') as f:
        pickle.dump(est, f, pickle.HIGHEST_PROTOCOL)

print('\n\nDone')
