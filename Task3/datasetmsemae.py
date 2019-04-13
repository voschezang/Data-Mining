'''
Run an experiment on any dataset obtained from the web, measure MSE and MAE of different
regression methods, and discuss the differences you find. 
(Make sure to include the link where you got the data from, add a sentence about why you chose that dataset,
 and another describing its size, attributes, etc.)
'''

import csv
import pandas as pd
import numpy as np
from dateutil.parser import parse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

boston = load_boston()
print(boston.data.shape)

regressor = DecisionTreeRegressor(random_state=0, criterion='mse')
cvs = cross_val_score(regressor, boston.data, boston.target, cv=5)
print(cvs)

regressor = DecisionTreeRegressor(random_state=0, criterion='mae')
cvs = cross_val_score(regressor, boston.data, boston.target, cv=)
print(cvs)