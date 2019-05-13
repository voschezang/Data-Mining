from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_squared_error
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pickle
import util.plot
import util.data
import numpy as np
from pandas.plotting import scatter_matrix
import sklearn.model_selection as ms


data = pd.read_csv('data/training_set_VU_DM_clean.csv', sep=';', nrows=10000)
columns = list(data.columns)
train, test = ms.train_test_split(data, random_state=42, test_size=0.3)
yLab = list(['click_bool', 'booking_bool', 'score', 'position'])
trainY = train['score']
trainYM = train[list(['click_bool', 'booking_bool'])]
trainX = train.drop(yLab, axis=1)
testY = test['score']
testYM = test[list(['click_bool', 'booking_bool'])]

testX = test.drop(yLab, axis=1)

np.random.seed(424242)

bools = [k for k in list(trainX.columns) if 'bool' in k]
isNulls = [k for k in list(trainX.columns) if 'is_null' in k]
weekDays = ['Friday',
            'Monday',
            'Saturday',
            'Sunday',
            'Thursday',
            'Tuesday']
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
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                max_depth=1, random_state=0, loss='ls').fit(trainX[varsUsed], trainY)
mean_squared_error(testY, est.predict(testX[varsUsed]))
est.feature_importances_

rfest = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rfest.fit(trainX[varsUsed], trainYM)
mean_squared_error(testYM, rfest.predict(testX[varsUsed]))
a2 = rfest.predict(testX[varsUsed])

rfReg = RandomForestRegressor(n_estimators=100)
rfReg.fit(trainX[varsUsed], trainY)
a3 = rfReg.predict(testX[varsUsed])
#predMat = pd.DataFrame()
