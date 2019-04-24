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


ids = [k for k in data.columns if 'id' in k]
for k in ids:
    util.data.replace_missing(data_clean, k)
    util.data.clean_id(data_clean, k)
    util.data.discretize(data_clean, k, Encoders)
    categorical.append(k)


print('\n--------')
print('Done')
