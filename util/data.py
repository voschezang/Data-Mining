
from termcolor import colored
from sklearn import preprocessing
import collections
import numpy as np
import pandas as pd
np.random.seed(123)


def print_warning(*args):
    print(colored(*args, 'red'))


def print_primary(*args):
    print(colored(*args, 'green'))


def print_secondary(*args):
    print(colored(*args, 'blue'))


def count_null_values(data, k):
    return data[k].isnull().sum()


def normalize(data, k, strict=False):
    print('\tnormalize row')
    #     data[[k]].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    #     data[[k]].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
    x = data[k]
    if strict:
        # minmax, i.e. zero mean and unit range
        data[k] = (x - x.mean()) / (x.max() - x.min())
    else:
        # z-score, i.e. zero mean and unit std dev
        # z-score is less dependent on outliers
        data[k] = (x - x.mean()) / x.std()


def log_normalize(data, k):
    print('\tlog-normalize row')
    x = data[k]
    # data[k] = np.log10(x)
    if np.any(x < 0):
        print_warning('\tSigned log')
    data[k] = np.sign(x) * np.log10(x.abs() + 1e-9)


def replace_missing(data, k):
    row = data[k]
    n = count_null_values(data, k)
    if n > 0:
        n_rel = n / data.shape[0] * 100
        print('\tReplace %i null values (%0.2f%%)' % (n, n_rel))
    # note that pd.where different than np.where
#     avg = np.nanmedian([x for x in X])
#     row = np.where(row.isnan(), row.median(), row)
    # row.where(row.notna(), row.median(), inplace=True)
    row.fillna(row.median(), inplace=True)


def clean_id(data, k):
    """ Clean numerical field `k` that represents an idea
    """
    print_primary('\nclean id in `%s`' % k)
    assert data[k].isnull().sum() == 0, 'Missing values shoud be removed'
    row = data[k]
    n = data[k].unique().size
    n_max = 9
    if n > n_max:
        print('\tCombine values')
        if row.dtype == 'int64':
            other = max(data[k].unique()) + 1
            if other < 1e16:
                other = 1e16
        else:
            other = 'Other'
        most_common = select_most_common(row, n=n_max - 1, key=other)
        keys = most_common.keys()
        row.where(row.isin(keys), other, inplace=True)


def flag_null_values(data, k):
    # add attribute to indicate null-values (i.e. 1 if null otherwise 0)
    k_new = k + '_is_null'
    print('\tFlag null values (adding attr `%s`)' % k_new)
    data[k_new] = 0
    # .where replaces locations where condition is False
    data[k_new].where(data[k].notna(), 1, inplace=False)


def clean_star_rating(data, k):
    print_primary('\nclean star rating: `%s`' % k)
    if count_null_values(data, k) / data.shape[0] > 0.05:
        flag_null_values(data, k)

    normalize(data, k)
    replace_missing(data, k)


def clean_usd(data, k):
    print_primary('\nclean usd: `%s`' % k)
    log_normalize(data, k)
    normalize(data, k)


def discretize(data, k, E, n_bins=None):
    """ Encode data[k] to a numerical format (in range [0,n_bins])
    Use stragegy=`uniform` when encoding integers (e.g. id's)

    :E Encoder object with attributes `encoders`, `decoders`
    """
    print_secondary('dicretize `%s`' % k)
    if n_bins is None:
        n_bins = data[k].unique().size
    X = data[k]
    X = np.array([x for x in X]).reshape(-1, 1)
    bins = np.repeat(n_bins, X.shape[1])  # e.g. [5,3] for 2 features
    # encode to integers
    # quantile: each bin contains approx. the same number of features
    strategy = 'uniform' if data[k].dtype == 'int64' else 'quantile'
    est = preprocessing.KBinsDiscretizer(
        n_bins=bins, encode='ordinal', strategy=strategy)
    est.fit(X)
    n_bins = est.bin_edges_[0].size
    s = ''
    if data[k].dtype == 'int64':
        print('\tAttribute & Number of bins (categories)')
        print('\t%s & %i \\\\' % (strategy, n_bins))
    else:
        print('\tbins (%i, %s):' % (n_bins, strategy))
        print('\tAttribute & Bin start & Bin 1 & Bin 2 \\\\')
        for st in [round(a, 3) for a in est.bin_edges_[0]]:
            s += '$%s$ & ' % str(st)
        print('\t\t%s & %s\n' % (k, s[:-2]))

#     data[k + ' bin'] = est.transform(X)
#     data.drop(k)
    data[k] = est.transform(X)
    E.discretizers[k] = est


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
