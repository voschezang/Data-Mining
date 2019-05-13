# from util.string import print_primary, print_secondary, print_warning
# from util import string
from sklearn import linear_model
import math
import pycountry
import requests
import iso3166
from phonenumbers.phonenumberutil import region_code_for_country_code
# from matplotlib import rcParams
import calendar
from dateutil.parser import parse
import pandas as pd
# from sklearn import preprocessing
import collections
import numpy as np


np.random.seed(123)


def count_null_values(data, k):
    return data[k].isnull().sum()


def proportion_null_values(data, k):
    return count_null_values(data, k) / data.shape[0]


def is_int(row: pd.Series) -> bool:
    return row.dtype == 'int64' or 'int' in str(row.dtype)


def join_inplace(data: pd.DataFrame, rows: np.ndarray, original_k: str, k_suffix='label'):
    for i in range(rows.shape[1]):
        data['%s_%s%i' % (original_k, k_suffix, i)] = rows[:, i]


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


def fix_labels(labels):
    # update in-place
    for i, v in enumerate(labels.copy()):
        if v is None:
            labels[i] = 'None'
        else:
            labels[i] = fix_label(labels[i])


def fix_label(x):
    max_length = 12
    x = str.title(str(x))
    if x is None:
        assert False
        return 'None'
    translations = {'Ja': 'Yes', 'Nee': 'No', '1': 'Yes', '0': 'No',
                    'I have no idea what you are talking about': 'Unknown'}
    if x in translations.keys():
        return translations[x]
    return x[:max_length]


def getDayName(weekday):
    return calendar.day_name[weekday]


def regress_booking(regData, fullK):
    X = regData[fullK]
    Y = regData['gross_bookings_usd']
    reg = linear_model.LinearRegression()
    reg.fit(X, Y)
    return reg


def times_to_string(times, *args, **kwargs):
    return [time_to_string(time, *args, **kwargs) for time in times]


def time_to_string(time, include_date=True):
    year_month_day = '%Y%m%d'
    hour_min_sec = '%H%M%S'
    if include_date:
        return parse(time).strftime(year_month_day + hour_min_sec)
    return parse(time).strftime(hour_min_sec)


def to_floats(X=[], default=0):
    try:
        return np.array(X).astype(float)
    except ValueError:
        #         if ':' in X[0]:
            # assume dates
        #             return times_to_string(X)
        for i, x in enumerate(X.copy()):
            try:
                if ':' in x:
                    x = x.replace(':', '')
                if ',' in x:
                    # TODO and not '.' in x
                    x = x.replace(',', '.')
                print(x)
                X[i] = float(x)
            except TypeError:
                X[i] = default
        return X


def filter_nans(x, y):
    x = np.array(x)
    y = np.array(y)
#     i_x = x[~np.isnan(x)]
    i_x = np.where(~np.isnan(x))
    x = x[i_x]
    y = y[i_x]
#     i_y = y[~np.isnan(y)]
    i_y = np.where(~np.isnan(y))
    x = x[i_y]
    y = y[i_y]
    return x, y

    # https://gis.stackexchange.com/questions/212796/get-lat-lon-extent-of-country-from-name-using-python


def get_boundingbox_country(country, output_as='boundingbox'):
    """
    get the bounding box of a country in EPSG4326 given a country name

    Parameters
    ----------
    country : str
        name of the country in english and lowercase
    output_as : 'str
        chose from 'boundingbox' or 'center'.
         - 'boundingbox' for [latmin, latmax, lonmin, lonmax]
         - 'center' for [latcenter, loncenter]

    Returns
    -------
    output : list
        list with coordinates as str
    """
    # create url
    output = -1
    url = '{0}{1}{2}'.format('http://nominatim.openstreetmap.org/search?country=',
                             country,
                             '&format=json&polygon=0')
    response = []
    w = 0
    while len(response) == 0:
        response = requests.get(url).json()
        if len(response) != 0:
            response = response[0]
        else:
            return output
        w += 1
        if w == 5:
            break
    # response = response[0]

    # parse response to list
    # if output_as == 'boundingbox':
    #     lst = response[output_as]
    #     output = [float(i) for i in lst]
    if output_as == 'center':
        lst = [response.get(key) for key in ['lat', 'lon']]
        output = [float(i) for i in lst]

    return output


def country_coordinates(country_id_numbers):
    '''
    Get and put country coordinates in a dictionary.
    '''
    countries_long_lat = {}

    for id in country_id_numbers.unique():
        id_region_code = region_code_for_country_code(id)
        id_str = str(id)
        if str(id) in iso3166.countries_by_numeric:
            # print("key in dict")
            id_region_code = iso3166.countries_by_numeric[str(id)][1]
        else:
            id_region_code = region_code_for_country_code(id)

        # 'ZZ' denotes 'unknown or unspecified country'
        if id_region_code == 'ZZ':
            countries_long_lat[id] = -1
            # countries_long_lat['ZZ'] = ''
            pass
        else:
            country_info = pycountry.countries.get(alpha_2=id_region_code)

            # get longitudal and latitudal coordinates of country
            ll = get_boundingbox_country(
                country=country_info.name, output_as='center')

            # key is the country id number
            countries_long_lat[id] = ll

    return countries_long_lat


def calculate_distance(a, b):
    '''
    Calculate the distance from point a to point b.
    Variables a and b are tuples containing a longitudal and a latitudal coördinate.
    '''

    # approximate radius of earth in km
    if a == -1 or b == -1:
        distance = np.nan
        return distance

    R = 6373.0

    # print("A", a)

    lata = math.radians(a[0])
    lona = math.radians(a[1])
    latb = math.radians(b[0])
    lonb = math.radians(b[1])

    distance_lon = lonb - lona
    distance_lat = latb - lata

    afs = math.sin(distance_lat / 2)**2 + math.cos(lata) * \
        math.cos(latb) * math.sin(distance_lon / 2)**2
    cir = 2 * math.atan2(math.sqrt(afs), math.sqrt(1 - afs))

    distance = R * cir

    # print("Result:", distance)
    return distance


def make_distance_matrix(countries_long_lat):
    '''
    Makes a distance matrix of all countries existing in the dictionary.
    Puts all keys with unknown
    '''
    n_o_countries = len(countries_long_lat.keys()) + 1
    key_number = 1
    distance_matrix = np.zeros((n_o_countries, n_o_countries))

    # assign rows and columns to keys
    for key in countries_long_lat:
        distance_matrix[key_number][0] = key
        distance_matrix[0][key_number] = key
        key_number += 1

    i = 1
    j = 1
    for key1 in countries_long_lat:
        i = 1
        for key2 in countries_long_lat:
            # if one of the keys is -1 no distance data is available
            if key1 == -1 or key2 == -1:
                distance_matrix[i][j] = np.nan
                i += 1

            # if the countries pointed by the key are the samen distance is 0
            elif key1 == key2:
                distance_matrix[i][j] = 0
                i += 1

            # else calculate distance from key1 to key2 and put in matrix
            else:
                c1 = countries_long_lat[key1]
                c2 = countries_long_lat[key2]
                distance_matrix[i][j] = calculate_distance(c1, c2)
                distance_matrix[j][i] = calculate_distance(c1, c2)
                i += 1
        j += 1

    # print(distance_matrix)
    return distance_matrix


def make_distance_dict(countries_long_lat):

    distances = {}
    for key1 in countries_long_lat:
        distances[key1] = {}

    for key1 in countries_long_lat:
        for key2 in countries_long_lat:
            c1 = countries_long_lat[key1]
            c2 = countries_long_lat[key2]

            if c1 == -1 or c2 == -1:
                distances[key1][key2] = np.nan

            # if the countries pointed by the key are the samen distance is 0
            elif c1 == c2:
                distances[key1][key2] = 0

            else:
                distances[key1][key2] = calculate_distance(c1, c2)

    return distances


# def construct_distance_attribute(country_id_numbers, country_id_destination):
#     travel_distances = []
#     for (id1, id2) in zip(country_id_numbers, country_id_destination):
#         distance = distances[id1][id2]
#         travel_distances.append(distance)
#         # print(id1, id2, distance)
#     return travel_distances


def long_lat_attr(data):
    country_id_numbers = data['prop_country_id']
    countries_long_lat = country_coordinates(country_id_numbers)
    lng = []
    lat = []
    for id in data['srch_id']:
        if id in countries_long_lat:
            lng.append(countries_long_lat[id][0])
            lat.append(countries_long_lat[id][1])
        else:
            lng.append(np.nan)
            lat.append(np.nan)

    data['longitudal'] = lng
    data['latitude'] = lat


def long_lat_attr(data):
    country_id_numbers = data['prop_country_id']
    countries_long_lat = country_coordinates(country_id_numbers)
    lng = []
    lat = []
    for id in data['srch_id']:
        if id in countries_long_lat:
            lng.append(countries_long_lat[id][0])
            lat.append(countries_long_lat[id][1])
        else:
            lng.append(np.nan)
            lat.append(np.nan)

    data['longitudal'] = lng
    data['latitude'] = lat


def calculate_DCG(rows):
    '''
    DCG = sum of all rows(gain / log2(rang in proposal lijst))
    NDCG = (gain / log2) / iDCG
    IDCG = ideal DCG = 3/log2 1 + 3/log2 2 + 3/log2 3

    Gains:
    5 - The user purchased a room at this hotel - booking bool true
    1 - The user clicked through to see more information on this hotel - click bool true
    0 - The user neither clicked on this hotel nor purchased a room at this hotel - both click and book not true
    '''
    DCG = 0
    for row in rows.itertuples(index=True, name='Pandas'):
        click_bool = getattr(row, 'click_bool')
        position = getattr(row, 'position')
        booking_bool = getattr(row, 'booking_bool')

        if booking_bool != 1:
            if position == 1:
                DCG += click_bool / position + 5 * booking_bool / position
            else:
                DCG += click_bool / \
                    math.log2(position) + 5 * booking_bool / \
                    math.log2(position)
        else:
            if position == 1:
                # perfect score
                DCG += click_bool / position + 5 * booking_bool / position
            else:
                DCG += 5 * booking_bool / math.log2(position)
    return DCG


def rows_srch_id(data, id):
    '''
    Get all rows of a single search id
    '''
    rows = data.loc[data['srch_id'] == id]
    return rows


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    # https://gist.github.com/bwhite/3726239
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    # https://gist.github.com/bwhite/3726239
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def relevance_scores(rows):
    positions = rows['position']
    p_max = int(positions.max()) + 1
    r = np.zeros(p_max)
    for row in rows.itertuples(index=True, name='Pandas'):
        click_bool = int(getattr(row, 'click_bool'))
        position = int(getattr(row, 'position'))
        booking_bool = getattr(row, 'booking_bool')
        if booking_bool > 0:
            r[position] = 5
        else:
            r[position] = 1 * click_bool
    return r


def NDCG_dict(data):
    NDCG = {}
    for id in data['srch_id'].unique():
        rows = rows_srch_id(data, id)
        r = relevance_scores(rows)
        ndcg = ndcg_at_k(r, r.size, method=0)
        NDCG[id] = ndcg
    return NDCG


def click_book_score(data):
    click_book_score = []
    for id in data['srch_id'].unique():
        rows = rows_srch_id(data, id)
        for row in rows.itertuples(index=True, name='Pandas'):
            click_bool = int(getattr(row, 'click_bool'))
            booking_bool = getattr(row, 'booking_bool')
            if booking_bool == 1:
                click_book_score.append(5)
            if click_bool == 1 and booking_bool != 1:
                click_book_score.append(1)
            elif booking_bool != 1 and click_bool != 1:
                click_book_score.append(0)
    return click_book_score
