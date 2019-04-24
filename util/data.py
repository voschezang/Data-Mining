
from sklearn import preprocessing
import collections
import numpy as np
import pandas as pd
np.random.seed(123)


def count_nans(data, k):
    return data[k].isnull().sum()


def replace_missing(data, k):
    row = data[k]
    n = count_nans(data, k)
    if n > 0:
        print('\tReplace %i null values' % n)
    # note that pd.where different than np.where
#     avg = np.nanmedian([x for x in X])
#     row = np.where(row.isnan(), row.median(), row)
    row.where(~row.isnull(), row.median(), inplace=True)
    # return row


def clean_id(data, k):
    print('\nclean id in `%s`' % k)
    assert data[k].isnull().sum() == 0, 'Missing values shoud be removed'
    row = data[k].copy()
    n = data[k].unique().size
    n_max = 10
    if n > n_max - 1 or True:
        print('\tCombine values')
        if row.dtype == 'int64':
            other = max(data[k].unique()) + 1
            if other < 1e16:
                other = 1e16
        else:
            other = 'Other'
        most_common = select_most_common(row, n=n_max, key=other)
        keys = most_common.keys()
        row.where(row.isin(keys), other, inplace=True)
    # return row


def discretize(data, k, E, n_bins=None):
    """
    :E Encoder object with attributes `encoders`, `decoders`
    """
    print('dicretize `%s`' % k)
    if n_bins is None:
        n_bins = data[k].unique().size
    X = data[k]
    X = np.array([x for x in X]).reshape(-1, 1)
    bins = np.repeat(n_bins, X.shape[1])  # e.g. [5,3] for 2 features
    # encode to integers
    # quantile: each bin contains approx. the same number of features
    est = preprocessing.KBinsDiscretizer(
        n_bins=bins, encode='ordinal', strategy='quantile')
    est.fit(X)
#     data[k + ' bin'] = est.transform(X)
#     data.drop(k)
    data[k] = est.transform(X)
    E.discretizers[k] = est
    s = ''
    print('\tbins (%i):' % n_bins)
    for st in [round(a, 3) for a in est.bin_edges_[0]]:
        #         if k == 'Year':
        #             st = int(round(st))
        s += str(st) + ', '
    print('\t\t\\textit{%s}: $\\{%s\\}$\n' % (k, s[:-2]))


def select_most_common(data: pd.Series, n=9, key="Other") -> dict:
    """ Return a dict containing the `n` most common keys and their count.
    :key the name of the new attribute that will replace the minority attributes
    """
    counts = collections.Counter(data)
    most_common = dict(counts.most_common(n))
    least_common = counts
    for k in most_common.keys():
        least_common.pop(k)

    most_common[key] = sum(least_common.values())
    return most_common


def summarize_categorical(data, k_x, k_y, conditional_x=False):
    # return the averages of each pair of categories
    # conditional_x means: P(Y|X=x) i.e. the distribution of Y given the value of X
    categories_x = data[k_x].unique()
    # categories_y = data[k_y].unique()
    # init dict
    summary = collections.defaultdict(dict)
    # fill dict
    for c_x in categories_x:
        view = data.loc[data[k_x] == c_x, k_y]
        # TODO assert that labels are beautified
        summary[fix_label(c_x)] = view.value_counts(
            normalize=conditional_x)  # average

    if not conditional_x:
        n = sum([a.sum() for a in summary.values()])
        for k_x in summary.keys():
            summary[k_x] /= n
    return summary


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

# parse(data.Bedtime[0]).strftime('%M%m%d'), data.Bedtime[0]
# parse(data.Bedtime[0]).strftime(year_month_day + hour_min_sec), data.Bedtime[0]
# def format_data(data):
#     if ':' in data[0]:
#         return times_to_string(data)
#     return data


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
