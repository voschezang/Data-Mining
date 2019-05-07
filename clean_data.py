import pandas as pd
from matplotlib import rcParams
import pickle
import util.plot
import util.data
import numpy as np
np.random.seed(123)


# import copy
# import scipy.stats
# import collections
# import seaborn as sns
# from matplotlib.ticker import NullFormatter
# import matplotlib.pyplot as plt
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
# rcParams['text.usetex'] = True
# import sklearn
# from dateutil.parser import parse
#data = pd.read_csv('data/training_set_VU_DM.csv', sep=',')

data = pd.read_csv('data/training_set_VU_DM.csv', sep=',', nrows=10000)
testdata=pd.read_csv('data/test_set_VU_DM.csv', sep=',', nrows=1000)
# data = pd.read_csv('data/training_set_VU_DM.csv', sep=',', nrows=100000)
# data.columns.sort_values()


E = util.data.Encoders()
E.discretizers = {}
E.encoders = {}

columns = list(data.columns)
# id
keys = [k for k in columns if 'id' in k]
for k in keys:
    util.data.replace_missing(data, k)
    util.data.clean_id(data, k)
    util.data.discretize(data, k, E)
util.string.remove(columns, keys)

# star ratings
data['delta_starrating'] = data['prop_starrating'] - data['visitor_hist_starrating']
columns.append('delta_starrating')
data = data.join(pd.get_dummies(data['prop_starrating'], 'prop_starrating_bool'))
data = data.join(pd.get_dummies(data['visitor_hist_starrating'], 'hist_starrating_bool'))
keys = [k for k in columns if 'starrating' in k]
for k in keys:
    util.data.clean_star_rating(data, k)
util.string.remove(columns, keys)

# float
data["srch_person_per_room_score"] = (
    data["srch_adults_count"] + data["srch_children_count"]) / data["srch_room_count"]
data.loc[~(data["srch_person_per_room_score"] < 10000),
         "srch_person_per_room_score"] = 0
data["srch_adults_per_room_score"] = data["srch_adults_count"] / \
    data["srch_room_count"]
data = data.join(pd.get_dummies(data['srch_adults_count'], 'srch_adults_count_bool'))
data = data.join(pd.get_dummies(data['srch_room_count'], 'srch_room_count_bool'))
data = data.join(pd.get_dummies(data['srch_children_count'], 'srch_children_count_bool'))
data = data.join(pd.get_dummies(data['prop_review_score'], 'prop_review_score_bool'))
keys = [k for k in columns if 'score' in k]
for k in keys:
    util.data.clean_float(data, k)
util.string.remove(columns, keys)

# categorical intss
# add attributes (bins) for each category of each categorical attr.
# i.e. encode categories in an explicit format
keys = util.string.select_if_contains(
    columns, ['count', 'position', 'srch_length_of_stay', 'srch_booking_window'])
for k in keys:
    util.data.clean_int(data, k, E)
util.string.remove(columns, keys)

# flag
keys = [k for k in columns if 'flag' in k]
for k in keys:
    util.data.print_primary(k)
    util.data.replace_missing(data, k)
util.string.remove(columns, keys)

# boolean
keys = [k for k in columns if 'bool' in k]
for k in keys:
    # TODO
    pass
util.string.remove(columns, keys)

# comp
keys = [k for k in columns if 'comp' in k]
compList = [k for k in keys if 'rate' in k and 'diff' not in k]
compDiffList = [k for k in keys if 'rate' in k and 'diff' in k]
availkeys = [k for k in keys if 'inv' in k]
data['unavailable_comp'] = data[availkeys].sum(axis=1)
availCompData = 1-data[availkeys]
data['available_comp'] = availCompData.sum(axis=1)
priceLevels = data[compDiffList] 
for k in range(0,len(compList)):
    priceLevels[compDiffList[k]] = priceLevels[compDiffList[k]] * data[compList[k]]
avgPriceLevel = priceLevels.mean(axis=1)
avgPriceLevel[avgPriceLevel.isna()] = 0
data['avg_price_comp'] = avgPriceLevel
    #TODO: outlier removal?
util.string.remove(columns, keys)


# date_time
k = 'date_time'
data = util.data.clean_date_time(data, k)
assert 'Monday' in data.columns
columns.remove(k)

# prop_log_historical_price
k = 'prop_log_historical_price'
data['has_historical_price'] = ~data['prop_log_historical_price'].isnull()
util.data.replace_missing(data, k, 0)
util.data.discretize(data, k, E, n_bins=3)
columns.remove(k)


# orig_destin_distance
k = 'orig_destination_distance'
util.data.replace_missing(data, k)
util.data.normalize(data, k)
columns.remove(k)


# usd
keys = [k for k in columns if 'usd' in k]
keys = keys[0:2]
data['has_purch_hist_bool'] = ~data['visitor_hist_adr_usd'].isnull()
for k in keys:
    util.data.clean_usd(data, k)
util.string.remove(columns, keys)


cols = regData.columns
keys1 = [k for k in cols if 'bool' in k] 
keys2 = [k for k in cols if 'null' in k]
keys3 = [k for k in cols if 'able_comp'in k]
keys4 = [k for k in cols if 'location_score' in k]
keys5 = [k for k in cols if 'prop_log' in k]
fullK = keys1+keys2+keys3+keys4+keys5  + ['avg_price_comp']
fullK.remove('booking_bool')
fullK.remove('click_bool')
regData = data.loc[~data['gross_bookings_usd'].isnull(),:]
bookingPred = util.data.regress_booking(regData,fullK)
data.loc[data['gross_bookings_usd'].isnull(),'gross_bookings_usd'] = bookingPred.predict(data.loc[data['gross_bookings_usd'].isnull(),fullK])

util.data.clean_usd(data, 'gross_bookings_usd')
util.string.remove('gross_bookings_usd')

# add score
data['score'] = data['click_bool'] + 5 * data['booking_bool']

print(len(columns), 'remaining attrs')
print(columns)

# add travel distance attribute
travel_distances = util.data.attr_travel_distances(data)
data['travel_distances'] = travel_distances

# save data & encoders
data.to_csv('data/training_set_VU_DM_clean.csv', sep=';', index=False)
with open('data/encoder.pkl', 'wb') as f:
    pickle.dump(E, f, pickle.HIGHEST_PROTOCOL)

print('\n--------')
print('Done')
