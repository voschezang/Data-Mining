'''
Run an experiment on any dataset obtained from the web, measure MSE and MAE of different
regression methods, and discuss the differences you find. 
(Make sure to include the link where you got the data from, add a sentence about why you chose that dataset,
 and another describing its size, attributes, etc.)
'''

# https://opendata.cbs.nl/statline/#/CBS/en/dataset/37259eng/table?dl=C09F
# dataset with population growth of Amsterdam
import csv
import pandas as pd
import numpy as np
from dateutil.parser import parse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# info = pd.read_csv('pdams.csv', sep=';', error_bad_lines=False)
info = pd.read_csv('pdams2.csv', sep=';')


# print(info.columns.values)
# print(info['Periods'], info['Population on 1 January (number)'])

periods = info['Periods'][7:14]
population_ams = info['Population on 1 January (number)'][7:14]

def func(x, a, b, c):
	return a*x**3+b*x+c

x = np.arange(0, len(periods), 1)
popt, pcov = curve_fit(func, x, population_ams)

y = []
for i in x:
	y.append(func(i, popt[0], popt[1], popt[2]))

# plt.plot(periods, y)
# plt.scatter(periods, population_ams)
# plt.show()

y_true = population_ams
y_pred = y

mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')


print(mse)
print(mae)
