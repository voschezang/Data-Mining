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
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import sklearn
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

boston = load_boston()
pd_bos = pd.DataFrame(boston.data)
print(boston.feature_names)
pd_bos.columns = boston.feature_names
pd_bos['PRICE'] = boston.target
X = pd_bos.drop('PRICE', axis=1)
y = pd_bos.PRICE

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
X2 = pd_bos[['CRIM', 'ZN']].copy()
X2_train, X2_test, Y2_train, Y2_test = sklearn.model_selection.train_test_split(X2, y, test_size=0.25, random_state=5)

regr_1 = DecisionTreeRegressor(criterion='mse')
regr_2 = DecisionTreeRegressor(criterion='mae')

model1 = regr_1.fit(X_train, Y_train)
model2 = regr_2.fit(X_train, Y_train)

# predict y values of test data
y_model1_predict = model1.predict(X_test)
y_model2_predict = model2.predict(X_test)

# R^2 score function
score1 = model1.score(X_test, Y_test)
score2 = model2.score(X_test, Y_test)
print(score1, score2)

model3 = regr_1.fit(X2_train, Y2_train)
model4 = regr_2.fit(X2_train, Y2_train)

y_model3_predict = model3.predict(X2_test)
y_model4_predict = model4.predict(X2_test)

def plot_residual(x_val, y_val, color):
	plt.figure()
	plt.scatter(x_val, y_val, c=color, s=40, alpha=0.5)
	line = np.linspace(0, 75, 100)
	plt.plot(line, line)
	plt.ylabel("Output")
	plt.xlabel("Input")
	plt.show()

# plot_residual(y_model1_predict,  Y_test, 'b')
# plot_residual(y_model2_predict,  Y_test, 'r')
# plot_residual(y_model3_predict,  Y_test, 'g')
# plot_residual(y_model4_predict,  Y_test, 'y')

# plt.figure()
# plt.scatter(np.linspace(0,70,len(y_model3_predict)), y_model3_predict)
# plt.xlabel("linspace")
# plt.ylabel("predicted price")
# plt.show()

# plt.figure()
# plt.scatter(X2_test['CRIM'], y_model3_predict)
# plt.xlabel("Crimerate")
# plt.ylabel("predicted price")
# plt.show()

# residuals plot shows true y-value on x-axis, predicted value on y-axis 
# plt.figure()
# plt.scatter(y_model1_predict,  Y_test, c='b', s=40, alpha=0.5)
# plt.plot(line, line)
# plt.ylabel("Output")
# plt.xlabel("Input")
# plt.show()

# plt.figure()
# plt.scatter(y_model2_predict,  Y_test, c='r', s=40, alpha=0.5)
# plt.plot(line, line)
# plt.ylabel("Output")
# plt.xlabel("Input")
# plt.show()