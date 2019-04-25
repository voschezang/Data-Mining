import util.data
import util.plot
# import pickle
# import copy
# import scipy.stats
# import collections
# import seaborn as sns
from matplotlib import rcParams
# from matplotlib.ticker import NullFormatter
# import matplotlib.pyplot as plt
# import sklearn
import pandas as pd
# from dateutil.parser import parse
import numpy as np
np.random.seed(123)
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
# rcParams['text.usetex'] = True


# data = pd.read_csv('data/training_set_VU_DM.csv', sep=';')
data = pd.read_csv('data/training_set_VU_DM.csv', sep=',', nrows=1000)
# data.columns.sort_values()
data_clean = data
categorical = []


class Encoders:
    discretizers = {}
    encoders = {}


columns = data.columns
ids = [k for k in columns if 'id' in k]
for k in ids:
    util.data.replace_missing(data_clean, k)
    util.data.clean_id(data_clean, k)
    util.data.discretize(data_clean, k, Encoders)
    categorical.append(k)

columns = [k for k in columns if k not in ids]
star_ratings = [k for k in columns if 'starrating' in k]
for k in star_ratings:
    util.data.clean_star_rating(data_clean, k)
    categorical.append(k)

columns = [k for k in columns if k not in star_ratings]
# k = 'visitor_hist_adr_usd'
usds = [k for k in columns if 'usd' in k]
for k in usds:
    util.data.clean_usd(data, k)

print(len(columns), 'remaining')

data_clean.to_csv('data/training_set_VU_DM_clean.csv', sep=';')
print('\n--------')
print('Done')
