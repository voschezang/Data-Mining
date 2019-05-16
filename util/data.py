from sklearn import linear_model
import math
import pycountry
import requests
import iso3166
from phonenumbers.phonenumberutil import region_code_for_country_code
import calendar
import pandas as pd
import collections
import numpy as np


np.random.seed(123)


def count_null_values(data, k):
    return data[k].isnull().sum()


def proportion_null_values(data, k):
    return count_null_values(data, k) / data.shape[0]


def is_int(row: pd.Series) -> bool:
    return row.dtype == 'int64' or 'int' in str(row.dtype)


def join_inplace(data: pd.DataFrame, rows: np.ndarray, original_k: str, k_suffix='label', keys=None):
    # k_suffix = str or list of str
    if isinstance(rows, np.ndarray):
        for i in range(rows.shape[1]):
            if keys is None:
                data['%s_%s%i' % (original_k, k_suffix, i)] = rows[:, i]
            else:
                data['%s_%s' % (original_k, keys[i])] = rows[:, i]
    else:
        # assume sparse array
        # TODO use sparsity
        join_inplace(data, rows.toarray(), original_k, k_suffix)


def scores_df(data, user_func=None, item_func=None):
    """ Convert `data` to a format suitable for the the surprise lib
    Surprise requires the order item-user-score
    user corresponds to search_id (i.e. a person), item to property id

    user_func and item_func can be used to transform (group) user/item ids
    user_func :: (pd.Series, int) -> int
    item_func :: (pd.Series, int) -> int
    """
    scores = {'item': [], 'user': [], 'score': []}
    for i in range(data.shape[0]):
        row = data.iloc[i]
        user_id = row.srch_id
        item_id = row.prop_id
        if user_func:
            user_id = user_func(row, user_id)
        if item_func:
            item_id = item_func(row, item_id)

        scores['user'].append(user_id)
        scores['item'].append(item_id)
        scores['score'].append(row.score)
    return pd.DataFrame(scores)


def signed_log(x):
    return np.sign(x) * np.log10(x.abs() + 1e-9)


def select_most_common(data: pd.Series, n=9, key="Other", v=1) -> dict:
    """ Return a dict containing the `n` most common keys and their count.
    :key the name of the new attribute that will replace the minority attributes
    """
    counts = collections.Counter(data)
    most_common = dict(counts.most_common(n))
    least_common = counts
    for k in most_common.keys():
        least_common.pop(k)

    most_common[key] = sum(least_common.values())
    if v:
        print('\tCombine %i categories' % len(least_common.keys()))
    return most_common


def replace_uncommon(row: pd.Series, common_keys=[], other=''):
    # Replace all values that are not in `common_keys`
    return row.where(row.isin(common_keys), other, inplace=False)


def regress_booking(regData, fullK):
    X = regData[fullK]
    Y = regData['gross_bookings_usd']
    reg = linear_model.LinearRegression()
    reg.fit(X, Y)
    return reg


def click_book_score(data):
    return (data['click_bool'] + 5 * data['booking_bool']
            ).transform(lambda x: min(x, 5))
