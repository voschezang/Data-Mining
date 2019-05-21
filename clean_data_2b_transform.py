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

data_all = pd.read_csv(
    'data/training_set_VU_DM_clean.csv', sep=';', nrows=5 * 1000)
# data_all = pd.read_csv('data/training_set_VU_DM_clean.csv', sep=';')

n = 10
# n = 5
k_user = 'AffinityPropagation'
k_item = 'FeatureAgglomeration'
k_user_long = clustering.USER_KEY_PREFIX + k_user
k_item_long = clustering.ITEM_KEY_PREFIX + k_item
keys_search, keys_property, _, _ = clustering.init(data_all)
models_user = []
models_item = []


def load_user_model():
    for i in range(n):
        print(i, 'data/est_user_%i.pkl' % i)
        with open('data/est_user_%i.pkl' % i, 'rb') as f:
            models_user.append(pickle.load(f))
    model_user = clustering.VotingRegressor(
        [(str(i), reg) for i, reg in enumerate(models_user)])
    return model_user


def load_item_model():
    for i in range(n):
        print(i, 'data/est_item_%i.pkl' % i)
        with open('data/est_item_%i.pkl' % i, 'rb') as f:
            models_item.append(pickle.load(f))

    model_item = clustering.VotingRegressor(
        [(str(i), reg) for i, reg in enumerate(models_item)])
    return model_item


def gc_collect(ensemble_model: clustering.VotingRegressor):
    for i, _ in enumerate(ensemble_model.estimators):
        ensemble_model.estimators[i] = None
    ensemble_model = None
    gc.collect()


model_user = load_user_model()
print('Transform training data (users)')
# for indices in np.split(np.arange(data_all.shape[0])):
# ..
result = clustering.predict(data_all, {k_user: model_user}, keys_search,
                            'srch_id', clustering.USER_KEY_PREFIX)
# users = users[clustering.USER_KEY_PREFIX + k_user]
data_all[k_user_long] = result[k_user_long]
# data_all.loc[indices, ..] = ..
# enforce disallocation of memory
result = None
gc_collect(model_user)


model_item = load_item_model()
print('Transform training data (items)')
result = clustering.predict(data_all, {k_item: model_item}, keys_property,
                            'prop_id', clustering.ITEM_KEY_PREFIX)
# items = items[clustering.ITEM_KEY_PREFIX + k_item]
data_all[k_item_long] = result[k_item_long]
# enforce disallocation of memory
result = None
gc_collect(model_item)

print('\nsetup scores df (1)')
k_user = clustering.USER_KEY_PREFIX + k_user
k_item = clustering.ITEM_KEY_PREFIX + k_item

# for k in users.columns:
# util.data.replace_extremely_uncommon(users, k)
# for k in items.columns:
#     # util.data.replace_extremely_uncommon(items, k)
#     data_all.loc[items.index, k] = items[k]


print('setup scores df (2)')
scores_train = util.data.scores_df(data_all, k_user, k_item)
scores_train_ = Dataset.load_from_df(scores_train, Reader(rating_scale=(0, 5)))
trainset, _ = surprise.model_selection.train_test_split(
    scores_train_, test_size=1e-15, random_state=seed)
print('fit SVD')
model = SVD()
model.fit(trainset)


# fill in predicted training data
print('predict training data with SVD')
data_all['score_svd'] = pd.Series()
scores_pred = clustering.svd_predict(model, scores_train)
# # scores_pred
for i, row in data_all.iterrows():
    #     score_pred = scores_pred[row[k_user]][row[k_item]]
    data_all['score_svd'] = scores_pred[row[k_user]][row[k_item]]

for k in data_all.columns:
    if 'cluster' in k:
        data_all.drop(columns=[k], inplace=True)

print('save training data to disk')
data_all.to_csv('data/training_set_VU_DM_clean_2.csv',
                sep=';', index=False)
data_all = None
# clear memory
gc.collect()

# predict test data
print('\nPredict test data')
# data = pd.read_csv('data/test_set_VU_DM_clean.csv', sep=';', nrows=1 * 1000)
data = pd.read_csv('data/test_set_VU_DM_clean.csv', sep=';')
util.data.rm_na(data)

model_user = load_user_model()
users = model_user.predict(data[keys_search])
gc_collect(model_user)

model_item = load_item_model()
items = model_item.predict(data[keys_property])
gc_collect(model_item)

data['score_svd'] = pd.Series()
for i, row in data.iterrows():
    result = model.predict(str(items[i]), str(users[i]), verbose=0)
    data.loc[i, 'score_svd'] = result.est

for k in data.columns:
    if 'cluster' in k:
        data_all.drop(columns=[k], inplace=True)

print('save test data to disk')
data.to_csv('data/test_set_VU_DM_clean_2.csv', sep=';', index=False)
print('\nDone')
