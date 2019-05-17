import sklearn
from sklearn import linear_model
import pandas as pd
import collections
import numpy as np


np.random.seed(123)


# def y_true(data_test: pd.DataFrame):
#     # return ncdg of y_true
#     y_true = data_test[['srch_id', 'prop_id',
#                         'click_bool', 'booking_bool']].copy()
#     add_score(y_true)
#     add_position(y_true)
#     return y_true

def y_pred_multi(x_test: pd.DataFrame, y_preds=[], weights=[], **kwargs):
    assert y_preds.shape[0] == np.array(weights).size
    y_pred_mean = np.mean(
        [y_preds[i] * w for i, w in enumerate(weights)], axis=0)
    return Xy_pred(x_test, y_pred_mean, **kwargs)


def Xy_pred(x_test: pd.DataFrame, y_pred: np.ndarray, save=False):
    y = to_df(x_test, y_pred)
    y = y.sort_values(['srch_id', 'score'], ascending=[True, False])
    if save:
        save_y_pred(y)
    return y


def save_y_pred(y_pred):
    # assume y_pred is sorted
    # .rename(columns={'srch_id': 'SearchId', 'prop_id': 'PropertyId'}, inplace=False)
    y = y_pred[['srch_id', 'prop_id']]
    y.to_csv('data/y_pred_result.csv', sep=',', index=False)


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


def to_df(x_test: pd.DataFrame, y_pred: np.ndarray):
    # convert model prediction to df
    y = x_test[['srch_id', 'prop_id']].copy()
    y['score'] = y_pred
    return y


def add_score(data: pd.DataFrame):
    data['score'] = click_book_score(data)


def add_position(data: pd.DataFrame):
    ordered = data.sort_values(['srch_id', 'score'], ascending=[True, False])
    data['position'] = pd.Series()
    prev_srch_id = None
    for i in ordered.index:
        #         print(data.loc[i].srch_id, ordered.loc[i].srch_id)
        # compute position, restart at each new srch_id
        if prev_srch_id != data.loc[i].srch_id:
            prev_srch_id = data.loc[i].srch_id
            position = 1
        else:
            position += 1

        # save value
        data.loc[i, 'position'] = position


def split_xy(data: pd.DataFrame,
             y_labels=['click_bool', 'booking_bool', 'score'], selection=None):
    # Return a tuple X,y
    # of type pd.DataFrame, np.array (compatible with sklearn)
    x = data.drop(columns=y_labels)
    y = data['score'].values
    if selection is not None:
        return x.loc[selection], y[selection]
    return x, y


def cv_folds_for_sklearn(data: pd.DataFrame, n_cv_folds=5, resampling_ratio=1):
    # Return "An iterable yielding (train, test) splits as arrays of indices"
    # I.e. the arg for sklearn.model_selection.cross_val_score(_, cv=arg)
    # :bco_splits = list of tuple of dataframes: (bookings, clicks, others)
    ids = sklearn.utils.shuffle(data.srch_id.unique(), random_state=123)
    ids_per_fold = np.array_split(ids, n_cv_folds)
    data_splits = split_data_based_on_ids(data, ids_per_fold)
    bco_splits = [split_bookings_clicks_others(split) for split in data_splits]
    return cv_folds(bco_splits, resampling_ratio)


def cv_folds(bco_splits, resampling_ratio):
    """ Return "An iterable yielding (train, test) splits as arrays of indices"
    I.e. the arg for sklearn.model_selection.cross_val_score(_, cv=arg)

    :bco_splits = list of tuple of dataframes: (bookings, clicks, others)
    """
    folds = resample_bco_splits(bco_splits, resampling_ratio)
    # for each step, choose (n-1) train folds and 1 test fold
    n_folds = len(folds)
    cv_folds = []
    for i in range(n_folds):
        if n_folds > 1:
            fold_indices = np.delete(np.arange(n_folds), i)
            # select & concatenate folds[indices]
            indices_train = combine_folds(folds, fold_indices)
            indices_test = folds[i]
        else:
            indices_train = folds[i]
            indices_test = []
        cv_folds.append((indices_train, indices_test))

    return cv_folds


def split_data_based_on_ids(data, ids_selections):
    return [data.loc[data.srch_id.isin(
        srch_ids)] for srch_ids in ids_selections]


def split_bookings_clicks_others(data):
    bookings = data.query('booking_bool == 1')
    clicks = data.query('click_bool == 1 and booking_bool != 1')
    others = data.query('click_bool != 1')

    for i in bookings.index[:100]:
        assert i not in clicks.index
        assert i not in others.index
    for i in clicks.index[:100]:
        assert i not in bookings.index
        assert i not in others.index

    return bookings, clicks, others


def resample_bco_splits(bco_splits, ratio=1):
    """ Returns a list of of folds, where each fold contains indices
    Ratio determines the amount of over/under sampling.

    :bco_splits = list of tuple of dataframes: (bookings, clicks, others)
    :ratio = float in [0,1] ratio of n_min ~ n_max. 0 means undersampling of the
    largest class and 1 means oversampling of the smallest class
    """
    folds = []
    for bco in bco_splits:
        n_max = max([df.shape[0] for df in bco])
        n_min = min([df.shape[0] for df in bco])
        # interpolate
        n = int(np.interp(ratio, [0, 1], [n_min, n_max]))
        fold_indices = sample_bco(bco, n)
        folds.append(fold_indices)

    return folds


def sample_bco(datasets=[], size_per_sample=100):
    sample_indices = [np.random.choice(data.index, size_per_sample)
                      for data in datasets
                      ]
    return np.concatenate(sample_indices)


def combine_folds(folds, indices):
    return np.concatenate([folds[i] for i in indices])


def sort_set(data):
    # sort test set predicted scores from high scores to low scores
    data = data.sort_values(['srch_id', 'score'], ascending=[True, False])
    return data
