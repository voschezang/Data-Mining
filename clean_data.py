""" Data preprocessing / preperation
Create a new dataframe consisting solely of floats.

Non-categorical fields have zero mean and unit variance (or max)
As money-related fields are lowerbounded (by zero) they are log-normalized
Replace missing values by the median (instead of the mean) of the respective
field, because it is assumed that most distributions are skewed.
"""
from sklearn.pipeline import Pipeline
import estimator
# , ExtendAttributes
from estimator import Estimator, Imputer, LabelBinarizer, Discretizer, MinMaxScaler, RobustScaler
from extended_attributes import ExtendAttributes
import pandas as pd
import pickle
import util.plot
import util.data
import numpy as np
np.random.seed(123)

# data = pd.read_csv('data/training_set_VU_DM.csv', sep=',')
data = pd.read_csv('data/training_set_VU_DM.csv', sep=',', nrows=1 * 1000)
data_test = pd.read_csv('data/test_set_VU_DM.csv', sep=',', nrows=1000)
# data = pd.read_csv('data/training_set_VU_DM.csv', sep=',', nrows=100000)
# data.columns.sort_values()


E = util.data.Encoders()
E.discretizers = {}
E.encoders = {}

columns = list(data.columns)
steps = []  # list of sklearn estimators


# # combine attributes
# steps.append(ExtendAttributes(columns))
# # TODO OneHotEncode floats?
# # steps.append(MinMaxNormalizer(LabelBinarizer.srch_person_per_room_score))
# steps.append(MinMaxScaler(ExtendAttributes.srch_person_per_room_score))
# steps.append(MinMaxScaler(ExtendAttributes.srch_adults_per_room_score))
# steps.append(MinMaxScaler(ExtendAttributes.delta_starrating))
# steps.append(LabelBinarizer(ExtendAttributes.weekday))
# # alt: for k in ExtendAttributes: steps.append(MinMaxNormalizer(k))
# # TODO add MinMaxNormalizer for other keys?


# id
keys = [k for k in columns if 'id' in k]
keys.remove('srch_id')
keys.remove('prop_id')
for k in keys:
    steps.append(LabelBinarizer(k))
util.string.remove(columns, keys)


# star ratings


keys = [k for k in columns if 'starrating' in k and 'hist' not in k and
        not ExtendAttributes.delta_starrating == k]
for k in keys:
    # util.data.clean_star_rating(data, k)
    steps.append(LabelBinarizer(k))
util.string.remove(columns, keys)

keys = [k for k in columns if 'hist' in k or ExtendAttributes.delta_starrating == k]
# TODO


# float
keys = [k for k in columns if 'score' in k]
for k in keys:
    # util.data.clean_float(data, k)
    steps.append(Imputer(k))
    steps.append(RobustScaler(k))
util.string.remove(columns, keys)

# categorical intss
# add attributes (bins) for each category of each categorical attr.
# i.e. encode categories in an explicit format
keys = util.string.select_if_contains(
    columns, ['count', 'position', 'srch_length_of_stay', 'srch_booking_window'])
for k in keys:
    steps.append(Imputer(k))
    steps.append(RobustScaler(k))
    steps.append(LabelBinarizer(k))  # removes the original class
util.string.remove(columns, keys)

# flag
keys = [k for k in columns if 'flag' in k]
for k in keys:
    util.data.print_primary(k)
    # util.data.replace_missing(data, k)
    steps.append(Imputer(k))
util.string.remove(columns, keys)

# date_time
k = 'date_time'
data = util.data.clean_date_time(data, k)
assert 'Monday' in data.columns
columns.remove(k)

# prop_log_historical_price
k = 'prop_log_historical_price'
steps.append(Imputer(k))
steps.append(Discretizer(k))
columns.remove(k)


# orig_destin_distance
k = 'orig_destination_distance'
steps.append(Imputer(k))
steps.append(RobustScaler(k))
columns.remove(k)


# usd
# keys = [k for k in columns if 'usd' in k]
# keys = keys[0:2]
# for k in keys:
#     util.data.clean_usd(data, k)
# util.string.remove(columns, keys)


# regData = data.loc[~data['gross_bookings_usd'].isnull(), :]
# cols = regData.columns
# keys1 = [k for k in cols if 'bool' in k]
# keys2 = [k for k in cols if 'null' in k]
# keys3 = [k for k in cols if 'able_comp'in k]
# keys4 = [k for k in cols if 'location_score' in k]
# keys5 = [k for k in cols if 'prop_log' in k]
# fullK = keys1 + keys2 + keys3 + keys4 + keys5 + ['avg_price_comp']
# fullK.remove('booking_bool')
# fullK.remove('click_bool')
# bookingPred = util.data.regress_booking(regData, fullK)
# data.loc[data['gross_bookings_usd'].isnull(), 'gross_bookings_usd'] = bookingPred.predict(
#     data.loc[data['gross_bookings_usd'].isnull(), fullK])
#
# util.data.clean_usd(data, 'gross_bookings_usd')
#
# # add score
# data['score'] = data['click_bool'] + 5 * data['booking_bool']
#
# print(len(columns), 'remaining attrs')
# print(columns)
#
# # add travel distance attribute
# data['travel_distance'] = util.data.attr_travel_distances(data)


# To be used in CF matrix factorization (SVD)
# scores = util.data.scores_df(data)


estimator.fit_transform(steps, data)
estimator.transform(steps, data_test)


# save data & encoders
data.to_csv('data/training_set_VU_DM_clean.csv', sep=';', index=False)
# scores.to_csv('data/scores_train.csv', sep=';', index=False)
with open('data/encoder.pkl', 'wb') as f:
    pickle.dump(E, f, pickle.HIGHEST_PROTOCOL)

print('\n--------')
print('Done')
