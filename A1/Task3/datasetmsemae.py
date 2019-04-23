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
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

boston = load_boston()
pd_bos = pd.DataFrame(boston.data)
# print(boston.feature_names)
pd_bos.columns = boston.feature_names
pd_bos['PRICE'] = boston.target
X = pd_bos.drop('PRICE', axis=1)
y = pd_bos.PRICE

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
X2 = pd_bos[['RM', 'LSTAT']].copy()
X2_train, X2_test, Y2_train, Y2_test = sklearn.model_selection.train_test_split(X2, y, test_size=0.25, random_state=5)

# method 1 & method 2

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

def plot_residual(x_val, y_val, color):
	plt.figure()
	plt.scatter(x_val, y_val, c=color, s=40, alpha=0.5)
	plt.hlines(y=0, xmin=0, xmax=55, linestyles='dashed')
	plt.ylabel("Output")
	plt.xlabel("Input")
	plt.show()

# plot_residual(y_model1_predict,  Y_test-y_model1_predict, 'b')
# plot_residual(y_model2_predict,  Y_test-y_model2_predict, 'r')

tree_MSE = mean_squared_error(y_model1_predict, Y_test)
tree_MSE2 = mean_squared_error(y_model1_predict, Y_test)

tree_MAE = mean_absolute_error(y_model2_predict, Y_test)
tree_MAE2 = mean_absolute_error(y_model2_predict, Y_test)

# method 3

# try different regression methods on data set
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# model evaluation
Y_lin_model = lin_model.predict(X_test)
linmodel_MSE = mean_squared_error(Y_test, Y_lin_model)
linmodel_MAE = mean_absolute_error(Y_test, Y_lin_model)

# method 4 https://blog.goodaudience.com/linear-regression-on-the-boston-housing-data-set-d18c4ce4d0be
def normal_equation(x, y):
	'''
	Normal equation will calculate correct weights of features.
	'''
	return np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)

theta = normal_equation(X_train, Y_train)
predictions = np.dot(X_train, theta)
test_pred = np.dot(X_test, theta)
MSE3 = mean_squared_error(test_pred, Y_test)
MAE3 = mean_absolute_error(test_pred, Y_test)

# method 5 Support Vector Machine
clf = svm.SVR()
clf.fit(X_train, Y_train) 
Y_SVR = clf.predict(X_test)

MSE_SVR = mean_squared_error(Y_SVR, Y_test)
MAE_SVR = mean_absolute_error(Y_SVR, Y_test)

print("DecisionTreeRegressor with MSE & MAE criterion")
print(tree_MSE, tree_MAE)

print("Linear Regression with scikit function")
print(linmodel_MSE, linmodel_MAE)

print("manual Linear Regression")
print(MSE3, MAE3)

print("support vector regression")
print(MSE_SVR, MAE_SVR)
