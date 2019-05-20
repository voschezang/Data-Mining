""" All newly added attributes (unparameterized) in a single class
"""
import pandas as pd
import calendar
from util.estimator import Estimator
import util.data


class ExtendedAttributes():
    srch_person_per_room_score = "srch_person_per_room_score"
    srch_adults_per_room_score = "srch_adults_per_room_score"
    unavailable_comp = "unavailable_comp"
    available_comp = "available_comp"
    delta_starrating = "delta_starrating"
    visitor_hist_adr_usd_log = "visitor_hist_adr_usd_log"
    price_usd_log = "price_usd_log"
    weekday = "weekday"


class ExtendAttributes(Estimator):
    def __init__(self, columns):
        self.columns = columns
        self.k = None

    def extend(self, data):
        data[ExtendedAttributes.srch_person_per_room_score] = (
            data["srch_adults_count"] + data["srch_children_count"]) / \
            data["srch_room_count"]
        data.loc[data[ExtendedAttributes.srch_person_per_room_score] >= 10000,
                 "srch_person_per_room_score"] = 0
        # print(data[ExtendedAttributes.srch_person_per_room_score].isna().sum())
        data[ExtendedAttributes.srch_adults_per_room_score] = data["srch_adults_count"] / \
            data["srch_room_count"]
        data[ExtendedAttributes.delta_starrating] = data['prop_starrating'] - \
            data['visitor_hist_starrating']

        data[ExtendedAttributes.visitor_hist_adr_usd_log] =  \
            data['visitor_hist_adr_usd'].transform(util.data.signed_log)
        data[ExtendedAttributes.price_usd_log] = \
            data['price_usd'].transform(util.data.signed_log)

        data['has_purch_hist_bool'] = (
            ~data['visitor_hist_adr_usd'].isnull()).astype(int)
        data['has_historical_price'] = (
            ~data['prop_log_historical_price'].isnull()).astype(int)

        keys = [k for k in self.columns if 'comp' in k]
        compList = [k for k in keys if 'rate' in k and 'diff' not in k]
        compDiffList = [k for k in keys if 'rate' in k and 'diff' in k]
        availkeys = [k for k in keys if 'inv' in k]
        data[ExtendedAttributes.unavailable_comp] = data[availkeys].sum(axis=1)
        availCompData = 1 - data[availkeys]
        data[ExtendedAttributes.available_comp] = availCompData.sum(axis=1)
        priceLevels = data[compDiffList]
        for k in range(0, len(compList)):
            # priceLevels[compDiffList[k]] == data[compDiffList[k]
            data[compDiffList[k]
                 ] = priceLevels[compDiffList[k]] * data[compList[k]]
        avgPriceLevel = priceLevels.mean(axis=1)
        avgPriceLevel[avgPriceLevel.isna()] = 0
        data['avg_price_comp'] = avgPriceLevel
        # data.drop(columns=[keys], inplace=True)

        # additional attributes
        datetimes = pd.to_datetime(data['date_time'])
        weekday = datetimes.dt.weekday  # 0 is monday
        weekday = [calendar.day_name[x] for x in weekday]
        year = datetimes.dt.year
        month = datetimes.dt.month
        day = datetimes.dt.day
        hour = datetimes.dt.hour
        minute = datetimes.dt.minute
        data['year'] = year
        data['month'] = month
        data['day'] = day
        data['hour'] = hour
        data['minute'] = minute
        data[ExtendedAttributes.weekday] = weekday
        data.drop(columns=['date_time'], inplace=True)

        # TODO add long, lat
        # add travel distance attribute
        # data['travel_distance'] = util.data.attr_travel_distances(data)
        # print(data['travel_distance'])

    def fit(self, data):
        data.drop(columns=['position'], inplace=True)
        data['score'] = util.data.click_book_score(data)
        # ignored keys will be imputed later
        keys_to_be_imputed = [ExtendedAttributes.delta_starrating,
                              ExtendedAttributes.srch_person_per_room_score,
                              ExtendedAttributes.visitor_hist_adr_usd_log,
                              ExtendedAttributes.price_usd_log,
                              'visitor_hist_starrating',
                              'orig_destination_distance'
                              ] + [k for k in data.columns if 'score' in k or 'usd' in k]
        util.data.rm_na(data, ignore=keys_to_be_imputed)

    def transform(self, data):
        pass
