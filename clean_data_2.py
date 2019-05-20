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
    'data/training_set_VU_DM_clean.csv', sep=';', nrows=10 * 1000)
# data_all = pd.read_csv('data/training_set_VU_DM_clean.csv', sep=';')


keys_search, keys_property, models_user, models_item = clustering.init(
    data_all)
# clustering.init_df_columns(data_all, models_user, models_item)
k_user = 'AffinityPropagation'
k_item = 'FeatureAgglomeration'
model = SVD()
model_user = models_user[k_user]
model_item = models_item[k_item]

# fit
clustering.fit(data_all, {k_user: model_user}, keys_search, 'srch_id')
clustering.fit(data_all, {k_item: model_item}, keys_property, 'prop_id')

users = clustering.predict(data_all, {k_user: model_user}, keys_search,
                           'srch_id', clustering.USER_KEY_PREFIX)
items = clustering.predict(data_all, {k_item: model_item}, keys_property,
                           'prop_id', clustering.ITEM_KEY_PREFIX)

for k in users.columns:
    util.data.replace_extremely_uncommon(users, k)
    data_all.loc[users.index, k] = users[k]
for k in items.columns:
    util.data.replace_extremely_uncommon(items, k)
    data_all.loc[items.index, k] = items[k]

k_user = clustering.USER_KEY_PREFIX + k_user
k_item = clustering.ITEM_KEY_PREFIX + k_item

scores_train = util.data.scores_df(data_all, k_user, k_item)
scores_train_ = Dataset.load_from_df(scores_train, Reader(rating_scale=(0, 5)))
trainset, _ = surprise.model_selection.train_test_split(
    scores_train_, test_size=1e-15, random_state=seed)
model.fit(trainset)


# fill in predicted training data
data_all['score_svd'] = pd.Series()
scores_pred = clustering.svd_predict(model, scores_train)
# # scores_pred
for i, row in data_all.iterrows():
    #     score_pred = scores_pred[row[k_user]][row[k_item]]
    data_all['score_svd'] = scores_pred[row[k_user]][row[k_item]]

for k in data_all.columns:
    if 'cluster' in k:
        data_all.drop(columns=[k], inplace=True)

data_all.to_csv('data/training_set_VU_DM_clean_2.csv', sep=';', index=False)
data_all = None
# clear memory
gc.collect()

# predict test data
data = pd.read_csv('data/test_set_VU_DM_clean.csv', sep=';', nrows=50 * 1000)
# data = pd.read_csv('data/test_set_VU_DM_clean.csv', sep=';')
util.data.rm_na(data)

for k in data.columns:
    assert not data[k].isna().any(), k
    assert not any(np.isinf(data[k]))

for i, row in data.iterrows():
    user = model_user.predict([row[keys_search]])
    item = model_item.predict([row[keys_property]])

data['score_svd'] = pd.Series()
for i, row in data.iterrows():
    result = model.predict(str(item), str(user), verbose=0)
    data.loc[i, 'score_svd'] = result.est


data.to_csv('data/test_set_VU_DM_clean_2.csv', sep=';', index=False)
print('\n\nDone')
