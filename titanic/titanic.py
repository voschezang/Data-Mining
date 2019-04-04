# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:16:42 2019

@author: Gillis
"""
import csv
import pandas as pd
import re
from dateutil.parser import parse
import numpy as np
import sklearn.model_selection as ms
import sklearn.ensemble as es
from sklearn import svm
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
from sklearn import linear_model
from matplotlib import rcParams
from sklearn.neural_network import MLPClassifier
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14

#Read in data, and prepare
dataF = pd.read_csv('C:\\Users\\Gillis\\Documents\\Uni\\Master2\\DMT\\Data-Mining\\titanic\\train.csv', sep=',')
#dataF = pd.read_csv('train.csv')
dataF = dataF.drop('Name',axis=1)
dataF["isMale"] = dataF["Sex"] == "male"
dataF = dataF.drop('Sex', axis=1)

# Fit regression for missing ages
dataF["isAgeImp"] = dataF["Age"].isna()
reg = linear_model.LinearRegression()
nonNaAge = dataF[dataF["Age"].notna()]
reg.fit(nonNaAge[["Fare","isMale","Parch","SibSp"]],nonNaAge["Age"])
agePreds = reg.predict(dataF[["Fare","isMale","Parch","SibSp"]])
dataF.loc[dataF["Age"].isna(),"Age"] = agePreds[dataF["Age"].isna()]

#show distibution of numeric vars
dataF["Age"].hist(rwidth=0.85,bins=20)
plt.savefig('ageHist.pdf')
dataF["Fare"].hist(rwidth=0.85,bins=20)
plt.savefig('fareHist.pdf')
dataF["SibSp"].hist(rwidth=0.85,bins=8)
dataF["Parch"].hist(rwidth=0.85,bins=6)

#Split categorical into binaries
dataF["isS"] = dataF["Embarked"] == "S"
dataF["isQ"] = dataF["Embarked"] == "Q"
dataF["lowClass"] = dataF["Pclass"]==1
dataF["highClass"] = dataF["Pclass"]==3
dataF= dataF.drop(["Embarked","Ticket","Pclass"],axis=1)
#dataF["highPrice"] = dataF["Fare"]>30
#dataF["lowPrice"] = dataF["Fare"]<10
for room in ["A","B","C","D","E","F"]:
    dataF[room] = dataF["Cabin"].str.contains(room)
    dataF.loc[dataF[room].isna(),room] = False
dataF= dataF.drop("Cabin",axis=1)
#dataF = dataF.drop("Fare",axis=1)

#standardise age
meanAge = dataF["Age"].mean()
stdAge = dataF["Age"].std()
dataF["Age"] =(dataF["Age"] - meanAge)/stdAge

#standardise price with logs
logP = np.log(dataF["Fare"]+0.1)
meanLogP = np.mean(logP)
stdLogP = np.std(logP)
dataF["logP"] = (logP-meanLogP)/stdLogP
#meanSibSp = dataF["SibSp"].mean()
#stdSibSp = dataF["SibSp"].std()
#dataF["SibSp"] =(dataF["SibSp"] - meanSibSp)/stdSibSp
#meanParch = dataF["Parch"].mean()
#stdParch = dataF["Parch"].std()
#dataF["Parch"] =(dataF["Parch"] - meanParch)/stdParch


dataF["family"] = dataF["Parch"] + dataF["SibSp"]

corrs = dataF.corr()
dataF = dataF.drop(["Parch","SibSp"],axis=1)

#Select variables
dataF = dataF.drop(["A","F","Age","isQ"], axis=1)
#sdataa = dataF
#dataF=sdataa
#dataF = dataF[["Survived","isMale","highClass","lowPrice","isS","B","highPrice","lowClass"]]

#Split train test for fitting models
train, test = ms.train_test_split(dataF, random_state = 42)

trainX = train.loc[:,"Age":]
trainY = train["Survived"]
testX = test.loc[:,"Age":]
testY = test["Survived"]
fullTrainX = dataF.loc[:,"Age":]
fullTrainY = dataF["Survived"]

#fit and train 3 models, select best
gbc_tuned_parameters = [{'loss':['deviance','exponential'],'learning_rate':[0.1,0.01,0.2],'n_estimators':[50,75,100,150,200],'max_depth':[1, 2, 3, 4],'min_samples_split' :[2,4]}]
gbc = ms.GridSearchCV(es.GradientBoostingClassifier(),gbc_tuned_parameters,cv=5,scoring='accuracy')
gbcScore = ms.cross_val_score(gbc, trainX, trainY, scoring='accuracy', cv=5)
print(np.mean(gbcScore))
gbc.fit(trainX,trainY)
print(gbc.score(testX,testY))
print(gbc.best_params_)
print(gbc.best_score_)

gbcModel = es.GradientBoostingClassifier(learning_rate = 0.2, loss='exponential',max_depth=3, min_samples_split=4, n_estimators=100)
gbcModel.fit(trainX,trainY)
print(gbcModel.score(testX,testY))
print(gbc.feature_importances_)

gbcModel.fit(trainX,trainY)
print(gbcModel.score(testX,testY))

tuned_parameters = [{'C':[0.01,0.1,1,10],'kernel':['linear','rbf'], 'gamma':['auto','scale']}]
svmModel = ms.GridSearchCV(svm.SVC(),tuned_parameters, scoring= 'accuracy', cv=5)
svmScore = ms.cross_val_score(svmModel, trainX, trainY, scoring='accuracy',cv=5)
print(np.mean(svmScore))

svmModel.fit(trainX,trainY)
print(svmModel.score(testX,testY))

tuned_parameters = [{'n_estimators':[10,50,75,100,150,200], 'max_depth':[1, 2, 3, 4, 5],'min_samples_split' :[2,4,6]}]
rfModel = ms.GridSearchCV(es.RandomForestClassifier(),tuned_parameters,cv=5,scoring='accuracy')
rfScore = ms.cross_val_score(rfModel, trainX, trainY, scoring='accuracy',cv=5)
print(np.mean(rfScore))

rfModel.fit(trainX,trainY)
print(rfModel.score(testX,testY))

tuned_parameters = [{'hidden_layer_sizes':[(5,2),(5),(10,2),(10,5),(4,2),(4,4),(5,5),np.arange(4,12)]}]
nnetModel = ms.GridSearchCV(MLPClassifier(max_iter=1000,solver='lbfgs'),tuned_parameters,cv=5,scoring='accuracy')
nnetScore = ms.cross_val_score(nnetModel, trainX, trainY, scoring='accuracy',cv=5)
print(np.mean(nnetScore))

nnetModel.fit(trainX,trainY)
print(nnetModel.score(testX,testY))

#train best final model
bestModel = es.GradientBoostingClassifier(learning_rate = 0.2, loss='exponential',max_depth=3, min_samples_split=4, n_estimators=100)
bestModel.fit(fullTrainX,fullTrainY)

#Redo data transformations for test set
dataT = pd.read_csv('C:\\Users\\Gillis\\Documents\\Uni\\Master2\\DMT\\Data-Mining\\titanic\\test.csv', sep=',')
dataT = dataT.drop('Name',axis=1)
dataT["isMale"] = dataT["Sex"] == "male"
dataT = dataT.drop('Sex', axis=1)

# Fit regression for missing ages
dataT["isAgeImp"] = dataT["Age"].isna()
#reg = linear_model.LinearRegression()
#nonNaAge = dataT[dataT["Age"].notna()]
#reg.fit(nonNaAge[["Fare","isMale","Parch","SibSp"]],nonNaAge["Age"])
agePreds = reg.predict(dataT[["Fare","isMale","Parch","SibSp"]])
dataT.loc[dataT["Age"].isna(),"Age"] = agePreds[dataT["Age"].isna()]

#Split categorical into binaries
dataT["isS"] = dataT["Embarked"] == "S"
dataT["isQ"] = dataT["Embarked"] == "Q"
dataT["lowClass"] = dataT["Pclass"]==1
dataT["highClass"] = dataT["Pclass"]==3
dataT= dataT.drop(["Embarked","Ticket","Pclass"],axis=1)
#dataT["highPrice"] = dataT["Fare"]>30
#dataT["lowPrice"] = dataT["Fare"]<10
for room in ["A","B","C","D","E","F"]:
    dataT[room] = dataT["Cabin"].str.contains(room)
    dataT.loc[dataT[room].isna(),room] = False
dataT= dataT.drop("Cabin",axis=1)
#dataT = dataT.drop("Fare",axis=1)

#standardise age
#meanAge = dataT["Age"].mean()
#stdAge = dataT["Age"].std()
dataT["Age"] =(dataT["Age"] - meanAge)/stdAge

#standardise price with logs
logP = np.log(dataT["Fare"]+0.1)
#meanLogP = np.mean(logP)
#stdLogP = np.std(logP)
dataT["logP"] = (logP-meanLogP)/stdLogP
#meanSibSp = dataT["SibSp"].mean()
#stdSibSp = dataT["SibSp"].std()
#dataT["SibSp"] =(dataT["SibSp"] - meanSibSp)/stdSibSp
#meanParch = dataT["Parch"].mean()
#stdParch = dataT["Parch"].std()
#dataT["Parch"] =(dataT["Parch"] - meanParch)/stdParch

dataT["family"] = dataT["Parch"] + dataT["SibSp"]

dataT = dataT.drop(["Parch","SibSp"],axis=1)

#Select variables
#sdataa = dataT
#dataT=sdataa
#dataT = dataT[["Survived","isMale","highClass","lowPrice","isS","B","highPrice","lowClass"]]

#predict test set and export
predicts = bestModel.predict(dataT.loc[:,"Age":])
finalDF = pd.DataFrame()
finalDF["PassengerId"] = dataT["PassengerId"]
finalDF["Survived"] = predicts
finalDF.to_csv('C:\\Users\\Gillis\\Documents\\Uni\\Master2\\DMT\\Data-Mining\\titanic\\titanicpredictions.csv', sep=',',index=False)
finalDF.to_csv('titanicpredictions.csv',sep=',',index=False)
