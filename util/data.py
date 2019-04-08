
import collections
import numpy as np
np.random.seed(123)


def summarize_categorical(data, k_x, k_y, conditional_x=False):
    # return the averages of each pair of categories
    # conditional_x means: P(Y|X=x) i.e. the distribution of Y given the value of X
    categories_x = data[k_x].unique()
    categories_y = data[k_y].unique()
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
