""" Data preprocessing / preperation
Create a new dataframe consisting solely of floats.

Non-categorical fields have zero mean and unit variance (or max)
As money-related fields are lowerbounded (by zero) they are log-normalized
Replace missing values by the median (instead of the mean) of the respective
field, because it is assumed that most distributions are skewed.
"""
# from sklearn.pipeline import Pipeline
# import estimator
# , ExtendAttributes
from pipeline import Pipeline
from estimator import Estimator, Imputer, LabelBinarizer, Discretizer, MinMaxScaler, RobustScaler, GrossBooking
from extended_attributes import ExtendedAttributes, ExtendAttributes
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


# combine attributes
# ExtendAttributes(columns)
steps.append(ExtendAttributes(columns))
steps.append(MinMaxScaler(ExtendedAttributes.srch_person_per_room_score))
steps.append(MinMaxScaler(ExtendedAttributes.srch_adults_per_room_score))
steps.append(MinMaxScaler(ExtendedAttributes.delta_starrating))
steps.append(Imputer(ExtendedAttributes.visitor_hist_adr_usd_log))
steps.append(MinMaxScaler(ExtendedAttributes.visitor_hist_adr_usd_log))
steps.append(Imputer(ExtendedAttributes.price_usd_log))
steps.append(MinMaxScaler(ExtendedAttributes.price_usd_log))
steps.append(LabelBinarizer(ExtendedAttributes.weekday))
# alt: for k in ExtendedAttributes: steps.append(MinMaxNormalizer(k))
# TODO add MinMaxNormalizer for other keys?

# # apply lin regression to attr before scaling dependent attrs
# k = 'gross_bookings_usd'
# steps.append(GrossBooking(k))
# columns.remove(k)


# id
keys = [k for k in columns if 'id' in k]
keys.remove('srch_id')
keys.remove('prop_id')
for k in keys:
    steps.append(LabelBinarizer(k))
util.string.remove(columns, keys)


# star ratings


keys = [k for k in columns if 'starrating' in k and 'hist' not in k
        and not ExtendedAttributes.delta_starrating == k]
for k in keys:
    # util.data.clean_star_rating(data, k)
    steps.append(LabelBinarizer(k))
util.string.remove(columns, keys)

keys = [k for k in columns if 'hist' in k or ExtendedAttributes.delta_starrating == k]
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
    columns, ['count', 'srch_length_of_stay', 'srch_booking_window'])
for k in keys:
    steps.append(Imputer(k))
    steps.append(RobustScaler(k))
    steps.append(LabelBinarizer(k))  # removes the original class
util.string.remove(columns, keys)

# flag
keys = [k for k in columns if 'flag' in k]
for k in keys:
    # util.data.print_primary(k)
    # util.data.replace_missing(data, k)
    steps.append(Imputer(k))
util.string.remove(columns, keys)

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
k = 'gross_bookings_usd'
steps.append(GrossBooking(k))
columns.remove(k)
# other usd
keys = [k for k in columns if 'usd' in k]
for k in keys:
    steps.append(Imputer(k))
    steps.append(RobustScaler(k))
util.string.remove(columns, keys)


print(len(columns), 'remaining attrs')  # TODO update this list
# print(columns)

pipeline = Pipeline(steps, data)
pipeline.transform(data_test)

# To be used in CF matrix factorization (SVD)
scores = util.data.scores_df(data)

# save data & encoders
data.to_csv('data/training_set_VU_DM_clean.csv', sep=';', index=False)
# scores.to_csv('data/scores_train.csv', sep=';', index=False)
with open('data/encoder.pkl', 'wb') as f:
    pickle.dump(E, f, pickle.HIGHEST_PROTOCOL)

print('\n--------')
print('Done')
