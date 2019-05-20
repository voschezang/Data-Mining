import pandas as pd
import numpy as np
from util import clustering
import util.data
seed = 123
np.random.seed(seed)


data = pd.read_csv('data/training_set_VU_DM_clean.csv',
                   sep=';', nrows=10 * 1000)
# TODO
# data = pd.read_csv('data/training_set_VU_DM_clean.csv', sep=';')

util.data.rm_na(data)
keys_search, keys_property, models_user, models_item = clustering.init(data)
clustering.train(data, keys_search, keys_property, models_user, models_item)

# transform test data
data = pd.read_csv('data/training_set_VU_DM_clean.csv', sep=';', nrows=1000)
# TODO
# data = pd.read_csv('data/training_set_VU_DM_clean.csv', sep=';')

clustering.transform(data, keys_search, keys_property,
                     models_user, models_item)
# TODO
# data.to_csv('data/test_set_VU_DM_clean.csv', sep=';', index=False)
