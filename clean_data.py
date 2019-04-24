import util.data
import util.plot
import pickle
import copy
import scipy.stats
import scipy.linalg
import collections
from IPython.display import HTML
import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import NullFormatter
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from dateutil.parser import parse
from importlib import reload
import numpy as np
np.random.seed(123)
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
# rcParams['text.usetex'] = True


# data = pd.read_csv('data/training_set_VU_DM.csv', sep=';')
data = pd.read_csv('data/training_set_VU_DM.csv', sep=',', nrows=1000)
