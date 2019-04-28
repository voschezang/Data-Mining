import pandas as pd
from matplotlib import rcParams
import pickle
import util.plot
import util.data
import numpy as np
np.random.seed(123)


# import copy
# import scipy.stats
# import collections
# import seaborn as sns
# from matplotlib.ticker import NullFormatter
# import matplotlib.pyplot as plt
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
# rcParams['text.usetex'] = True
# import sklearn
# from dateutil.parser import parse
# data = pd.read_csv('data/training_set_VU_DM.csv', sep=';')

data = pd.read_csv('data/training_set_VU_DM.csv', sep=',', nrows=1000)
# data.columns.sort_values()


class Encoders:
    discretizers = {}
    encoders = {}


columns = list(data.columns)
# id
keys = [k for k in columns if 'id' in k]
for k in keys:
    util.data.replace_missing(data, k)
    util.data.clean_id(data, k)
    util.data.discretize(data, k, Encoders)
util.string.remove(columns, keys)

# star ratings
keys = [k for k in columns if 'starrating' in k]
for k in keys:
    util.data.clean_star_rating(data, k)
util.string.remove(columns, keys)

# usd
keys = [k for k in columns if 'usd' in k]
for k in keys:
    util.data.clean_usd(data, k)
util.string.remove(columns, keys)

# float
keys = [k for k in columns if 'score' in k]
for k in keys:
    util.data.clean_float(data, k)
util.string.remove(columns, keys)

# categorical ints
keys = util.string.select_if_contains(
    columns, ['count', 'position', 'srch_length_of_stay', 'srch_booking_window'])
for k in keys:
    util.data.clean_int(data, k, Encoders)
util.string.remove(columns, keys)

# flag
keys = [k for k in columns if 'flag' in k]
for k in keys:
    util.data.print_primary(k)
    util.data.replace_missing(data, k)
util.string.remove(columns, keys)

# boolean
keys = [k for k in columns if 'bool' in k]
for k in keys:
    # TODO
    pass
util.string.remove(columns, keys)

# comp
keys = [k for k in columns if 'comp' in k]
for k in keys:
    # TODO
    pass
util.string.remove(columns, keys)


# date_time
k = 'date_time'
util.data.clean_date_time(data, k)
columns.remove(k)

# prop_log_historical_price
k = 'prop_log_historical_price'
util.data.replace_missing(data, k, 0)
util.data.discretize(data, k, Encoders, n_bins=3)
columns.remove(k)

print(len(columns), 'remaining attrs')
print(columns)

# save data & encoders
data.to_csv('data/training_set_VU_DM_clean.csv', sep=';')
fn = 'encoder.pkl'
with open(fn, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

print('\n--------')
print('Done')
