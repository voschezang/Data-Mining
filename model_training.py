from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_squared_error
import sklearn.model_selection as ms

import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pickle
import util.plot
import util.data
import numpy as np
from pandas.plotting import scatter_matrix
import sklearn.model_selection as ms


data = pd.read_csv('data/training_set_VU_DM_clean.csv', sep=';')
columns = list(data.columns)
srchIdList = np.unique(data['srch_id'])
trainID, testID = ms.train_test_split(srchIdList, random_state=42, test_size=0.4)
train = data[data['srch_id'].isin(trainID)]
test = data[data['srch_id'].isin(testID)]
#train, test = ms.train_test_split(data, random_state=42, test_size=0.3)
yLab = list(['click_bool', 'booking_bool', 'score', 'position'])
trainY = train[yLab]
trainX = train.drop(yLab, axis=1)
testY = test[yLab]

testX = test.drop(yLab, axis=1)

np.random.seed(424242)

bools = [k for k in list(trainX.columns) if 'bool' in k]
isNulls = [k for k in list(trainX.columns) if 'is_null' in k]
weekDays = ['Friday',
            'Monday',
            'Saturday',
            'Sunday',
            'Thursday',
            'Tuesday',
            'Other']
weekDays = ['weekday_' + k for k in weekDays]
comps = ['unavailable_comp',
         'available_comp',
         'avg_price_comp']
otherSelection = ['has_historical_price',
                  # 'travel_distances',
                  'delta_starrating',
                  'srch_query_affinity_score',
                  'price_usd',
                  'promotion_flag',
                  'prop_location_score1',
                  'prop_location_score2',
                  'srch_adults_per_room_score',
                  'srch_person_per_room_score']
dateTimes = ['day',
             'hour']
varsUsed = bools + isNulls + weekDays + comps + otherSelection
# varsUsed = ['visitor_hist_starrating',
#            'visitor_hist_adr_usd',
#            'prop_starrating',
#            'prop_review_score',
#            'prop_brand_bool',
#            'prop_location_score1',
#            'prop_location_score2',
#            'prop_log_historical_price',
#            'price_usd',
#            'promotion_flag',
#            'srch_length_of_stay',
#            'srch_booking_window',
#            'srch_adults_count',
#            'srch_children_count',
#            'srch_room_count',
#            'srch_saturday_night_bool',
#            'srch_query_affinity_score',
#            'visitor_hist_starrating_is_null',
#            'day',
#            'hour',
#            'srch_person_per_room',
#            'Friday',
#            'Monday',
#            'Saturday',
#            'Sunday',
#            'Thursday',
#            'Tuesday']

# start training models below

for k in varsUsed:
    if k not in trainX.columns:
        print(k)
        varsUsed.remove(k)

est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                max_depth=1, random_state=0, loss='ls').fit(trainX[varsUsed], trainY)
print(mean_squared_error(testY, est.predict(testX[varsUsed])))
est.feature_importances_

rfest = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rfest.fit(trainX[varsUsed], trainY[list(['click_bool', 'booking_bool'])])
a4 = rfest.predict(testX[varsUsed])
print(mean_squared_error(testY['click_bool']+5*testY['booking_bool'], a4[:,0]+5*a4[:,1] ))
a2 = rfest.predict(testX[varsUsed])
rfest_tuned_parameters = [{'max_depth':[1,2,4,5],'n_estimators':[50,100,150,200,300]}]
rfestGS = ms.GridSearchCV(RandomForestClassifier(),rfest_tuned_parameters,cv=5,scoring='neg_mean_squared_error')
rfestGS.fit(trainX[varsUsed],trainY[list(['click_bool', 'booking_bool'])])

rfReg = RandomForestRegressor(n_estimators=100)
rfReg.fit(trainX[varsUsed], trainY['score'])
a3 = rfReg.predict(testX[varsUsed])
print(mean_squared_error(testY['score'], est.predict(testX[varsUsed])))
rfreg_tuned_parameters = [{'max_depth':[1,2,4,5],'n_estimators':[50,100,150,200,300]}]
rfregGS = ms.GridSearchCV(RandomForestRegressor(),rfreg_tuned_parameters,cv=5,scoring='neg_mean_squared_error')
rfregGS.fit(trainX[varsUsed],trainY['score'])

adaReg = AdaBoostRegressor()
adaReg.fit(trainX[varsUsed],trainY['score'])
print(mean_squared_error(testY['score'],adaReg.predict(testX[varsUsed])))
ada_tuned_parameters = [{'loss':['linear','square'],'learning_rate':[0.5,1,2],'n_estimators':[50,100,150,25]}]
adaGS = ms.GridSearchCV(AdaBoostRegressor(),ada_tuned_parameters,cv=5,scoring='neg_mean_squared_error')
adaGS.fit(trainX[varsUsed],trainY['score'])
print(adaGS.score(testY['score'],adaGS.predict(testX[varsUsed])))
print(adaGS.best_params_)
print(adaGS.best_score_)
#predMat = pd.DataFrame()

rfReg = RandomForestRegressor(n_estimators=100)
rfReg.fit(trainX[varsUsed], trainY['score'])
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                max_depth=1, random_state=0, loss='ls').fit(trainX[varsUsed], trainY['score'])

adaTestX = testX[['srch_id','prop_id']]
adaTestY = testY
adaTestY['srch_id'] = testX['srch_id']
adaTestX['score'] = est.predict(testX[varsUsed])
# sort test set predicted scores from high scores to low scores
adaTestX = adaTestX.iloc[1:100000,:]
adaTestY = adaTestY.iloc[1:100000,:]

adaTestX = adaTestX.sort_values(['srch_id', 'score'], ascending=[True, False])

# create new column to store position of prop_id
adaTestX['position'] = pd.Series()

# save position prop_id
prev_srch_id = -1
for i in adaTestX.index.tolist():
    row = adaTestX.loc[i]
    # compute position
    if prev_srch_id != row.srch_id:
        position = 1
        prev_srch_id = row.srch_id
    else:
        position += 1
        
    # save position value to X_test
    adaTestX.loc[i, 'position'] = int(position)

    # to calculate the DCG of the test set old scores have to be used instead of the predicted scores
    adaTestX.loc[i, 'score'] = testY.loc[i, 'score']
    
# X_test
# determine ideal positions for test set with the 'real' scores, later the ideal DCG for the test set can be determined
adaTestY = adaTestY.sort_values(['srch_id', 'score'], ascending=[True, False])
adaTestY['position'] = pd.Series()
prev_srch_id = -1
for i in adaTestY.index.tolist():
    row = adaTestY.loc[i]
    # compute position
    if prev_srch_id != row.srch_id:
        position = 1
        prev_srch_id = row.srch_id
    else:
        position += 1

    # save value to X_test
    adaTestY.loc[i, 'position'] = int(position)

# calculate dcg of test set per srch_id
ndcg_test = util.data.DCG_dict(adaTestX)

# calculate ideal dcg of test set per srch_id
ndcg_control = util.data.DCG_dict(adaTestY)

# calculate means of both dcg dictionaries
print(np.mean(list(ndcg_test.values())))
print(np.mean(list(ndcg_control.values())))

# normalize
ndcg = np.mean(list(ndcg_test.values())) / np.mean(list(ndcg_control.values()))
ndcg


