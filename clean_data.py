""" Data preprocessing / preperation
Create a new dataframe consisting solely of floats.

Non-categorical fields have zero mean and unit variance (or max)
As money-related fields are lowerbounded (by zero) they are log-normalized
Replace missing values by the median (instead of the mean) of the respective
field, because it is assumed that most distributions are skewed.
"""
from util.pipeline import Pipeline
from util.estimator import Imputer, RemoveKey, LabelBinarizer, Discretizer, MinMaxScaler, RobustScaler, GrossBooking
from util.extended_attributes import ExtendedAttributes, ExtendAttributes
import pandas as pd
import util.plot
from util.string import print_primary
import gc
import util.data
import numpy as np
np.random.seed(123)

# data = pd.read_csv('data/training_set_VU_DM.csv', sep=',')
data = pd.read_csv('data/training_set_VU_DM.csv', sep=',', nrows=100 * 1000)
# data = data.to_sparse(fill_value=0)

columns = list(data.columns)
steps = []  # list of sklearn estimators

###############################################################################
# Build pipeline
###############################################################################

# combine attributes
# ExtendAttributes(columns)
steps.append(ExtendAttributes(columns))
steps.append(Imputer(ExtendedAttributes.srch_person_per_room_score))
steps.append(MinMaxScaler(ExtendedAttributes.srch_person_per_room_score))
steps.append(MinMaxScaler(ExtendedAttributes.srch_adults_per_room_score))
steps.append(Imputer(ExtendedAttributes.delta_starrating))
steps.append(MinMaxScaler(ExtendedAttributes.delta_starrating))
steps.append(Imputer(ExtendedAttributes.visitor_hist_adr_usd_log))
steps.append(MinMaxScaler(ExtendedAttributes.visitor_hist_adr_usd_log))
steps.append(Imputer(ExtendedAttributes.price_usd_log))
steps.append(MinMaxScaler(ExtendedAttributes.price_usd_log))
steps.append(LabelBinarizer(ExtendedAttributes.weekday, 5, use_keys=True))
steps.append(RemoveKey(ExtendedAttributes.weekday))


# id
keys = [k for k in columns if 'id' in k]
keys.remove('srch_id')
keys.remove('prop_id')
for k in keys:
    steps.append(LabelBinarizer(k))
    steps.append(RemoveKey(k))
util.string.remove(columns, keys)


# star ratings
keys = [k for k in columns if 'starrating' in k and 'hist' not in k
        and not ExtendedAttributes.delta_starrating == k]
for k in keys:
    steps.append(Imputer(k))
    steps.append(LabelBinarizer(k))
    steps.append(RemoveKey(k))
util.string.remove(columns, keys)

# float
keys = [k for k in columns if 'score' in k or k == 'visitor_hist_starrating']
for k in keys:
    # util.data.clean_float(data, k)
    steps.append(Imputer(k))
    steps.append(RobustScaler(k))
util.string.remove(columns, keys)

# ints
keys = util.string.select_if_contains(
    columns, ['count', 'srch_length_of_stay', 'srch_booking_window'])
for k in keys:
    steps.append(Imputer(k))
    # steps.append(LabelBinarizer(k))
    steps.append(Discretizer(k))
    steps.append(RobustScaler(k))
util.string.remove(columns, keys)

# flag
keys = [k for k in columns if 'flag' in k]
for k in keys:
    steps.append(Imputer(k))
util.string.remove(columns, keys)

k = 'prop_log_historical_price'
steps.append(Imputer(k))
steps.append(RobustScaler(k))
columns.remove(k)

k = 'orig_destination_distance'
steps.append(Imputer(k))
steps.append(Discretizer(k))
steps.append(RobustScaler(k))
# steps.append(MinMaxScaler(k))
# steps.append(RemoveKey(k))
columns.remove(k)

# usd
k = 'gross_bookings_usd'
steps.append(GrossBooking(k))
columns.remove(k)
# other usd
keys = [k for k in columns if 'usd' in k]
for k in keys:
    steps.append(Imputer(k))
    # steps.append(RobustScaler(k))
    steps.append(Discretizer(k))
    steps.append(RemoveKey(k))
util.string.remove(columns, keys)

# TODO???
# # add travel distance attribute
# data['travel_distance'] = util.data.attr_travel_distances(data)

# TODO mv to ExtendAttributes.fit()
# # add longitudal and latitudal coordinates of destination country
# lng, lat = util.data.attr_long_lat(data)
# data['longitudal'] = lng
# data['latitude'] = lat

# TODO???
# # relevance score per single search put in attribute
# data['relevance'] = util.data.click_book_score(data)


print(len(columns), 'remaining attrs')  # TODO update this list
# print(columns)

###############################################################################
# Apply pipeline
###############################################################################

print_primary('\n ----- \n Fit estimator models on training data \n ---- \n')
pipeline = Pipeline(steps, data)
# save data to disk
# with open('data/pipeline.pkl', 'wb') as f:
# pickle.dump(pipeline, f, pickle.HIGHEST_PROTOCOL)
data.to_csv('data/training_set_VU_DM_clean.csv', sep=';', index=False)
data = None
# clear memory
gc.collect()

# Transfrom test data
print_primary('\n\n ----- \n Transform test data \n ---- \n\n')
# data_test = pd.read_csv('data/test_set_VU_DM.csv', sep=',')
data_test = pd.read_csv('data/test_set_VU_DM.csv', sep=',', nrows=1000 * 1000)

pipeline.transform(data_test)
# data_test.to_csv('data/test_set_VU_DM_clean.csv', sep=';', index=False)
data_test.to_csv('data/test_set_VU_DM_clean.csv', sep=';', index=False)

print('\n--------')
print('Done')
