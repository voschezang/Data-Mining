""" All newly added attributes (unparameterized) in a single class
"""
from enum import Enum
import pandas as pd
from util.estimator import NoTransformEstimator
import util.data


class ExtendedAttributes(Enum):
    srch_person_per_room_score = "srch_person_per_room_score"
    srch_adults_per_room_score = "srch_adults_per_room_score"
    unavailable_comp = "unavailable_comp"
    available_comp = "available_comp"
    delta_starrating = "delta_starrating"
    visitor_hist_adr_usd_log = "visitor_hist_adr_usd_log"
    price_usd_log = "price_usd_log"
    weekday = "weekday "


class ExtendAttributes(NoTransformEstimator):
    def __init__(self, columns):
        self.columns = columns

    def extend(self, data):
        data[ExtendedAttributes.srch_person_per_room_score] = (
            data["srch_adults_count"] + data["srch_children_count"]) / \
            data["srch_room_count"]
        data.loc[data[ExtendedAttributes.srch_person_per_room_score] >= 10000,
                 "srch_person_per_room_score"] = 0
        data[ExtendedAttributes.srch_adults_per_room_score] = data["srch_adults_count"] / \
            data["srch_room_count"]
        data['has_historical_price'] = ~data['prop_log_historical_price'].isnull()
        data[ExtendedAttributes.delta_starrating] = data['prop_starrating'] - \
            data['visitor_hist_starrating']

        data[ExtendedAttributes.visitor_hist_adr_usd_log] =  \
            data['visitor_hist_adr_usd'].transform(util.data.signed_log)
        data[ExtendedAttributes.price_usd_log] = \
            data['price_usd'].transform(util.data.signed_log)

        data['has_purch_hist'] = ~data['visitor_hist_adr_usd'].isnull()

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

        # additional attributes
        datetimes = pd.to_datetime(data['date_time'])
        weekday = datetimes.dt.weekday  # 0 is monday
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
        # add score
        data['score'] = data['click_bool'] + 5 * data['booking_bool']
