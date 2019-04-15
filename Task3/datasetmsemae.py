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
scores = cross_val_score(regressor, boston.data, boston.target, cv=10)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

regressor = DecisionTreeRegressor(random_state=0, criterion='mae')
scores2 = cross_val_score(regressor, boston.data, boston.target, cv=10)
print(scores2)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))


from sklearn.datasets import load_boston
from sklearn.linear_model import RidgeCV
from sklearn.cross_validation import cross_val_score

boston = load_boston()
# ridgeCV 
mean = np.mean(cross_val_score(RidgeCV(), boston.data, boston.target, scoring='mean_squared_error'))
mean2 = np.mean(cross_val_score(RidgeCV(), boston.data, boston.target, scoring='mean_absolute_error'))
print(mean, mean2)