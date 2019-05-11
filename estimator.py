from sklearn.base import TransformerMixin
import pandas as pd
from sklearn import preprocessing
import util.data
from util.string import print_primary, print_secondary, print_warning

# Pipeline of Estimators


def fit_transform(steps, data: pd.DataFrame):
    # steps : iterable of Estimator
    for est in steps:
        est.fit(data)
        est.transform(data)


def transform(steps, data: pd.DataFrame):
    # steps : iterable of Estimator
    for est in steps:
        est.transform(data)

# Estimator


class Estimator:
    # Interface, partly compatible with sklearn.base.TransformerMixin
    def __init__(self, k: str = ''):
        self.k = k
        self.est = None

    def fit(self, data: pd.DataFrame):
        pass

    def transform(self, data):
        raise NotImplementedError


class Dummy(Estimator):
    # def fit(self, data):
    #     pass

    def transform(self, data):
        pass


class Imputer(Estimator):
    """ Imputer class for pandas.DataFrame
    .tranform replaces missing values by median
    """

    def fit(self, data):
        self.value = data[self.k].median()

    def transform(self, data):
        data[self.k].fillna(self.value, inplace=True)


class LabelBinarizer(Estimator):
    """ Wrapper for sklearn.preprocessing.LabelBinarizer, allowing
    pandas.DataFrame mutations
    Can be used to one-hot encode categorical labels
    """

    def __init__(self, k):
        super().__init__(k)
        self.n_max = 10  # TODO use larger number

    def fit(self, data):
        """ Find most common catorgies, group uncommon categories under a
        single `uncommon_value` and fit estimator
        """
        print_primary('\nOneHotEncode labels `%s`' % self.k)
        # max_prop_null_values = 0.05
        # if util.data.proportion_null_values(data, self.k) > max_prop_null_values :
        # flag_null_values(data, self.k)

        if data[self.k].dtype == 'int64' or 'int' in str(data[self.k].dtype):
            # TODO or any other int
            self.na_value = data[self.k].max() + 1
        else:
            if 'float' in str(data[self.k].dtype):
                print_warning('\t Using LabelBinarizer for floats')
            # assume dtype == string
            self.na_value = '_pd_na'

        # assert data[self.k].isnull().sum(
        # ) == 0, 'Missing values shoud be removed'
        # row = data[self.k].copy()
        row = self._transform_na(data)
        # add a new category to group uncommen categories
        if row.dtype == 'int64':
            self.uncommon_value = max(row.unique()) + 1
            self.uncommon_value = max([1e16, self.uncommon_value + 1])
        else:
            self.uncommon_value = 'Other'

        most_common = util.data.select_most_common(
            row, n=self.n_max - 1, key=self.uncommon_value, v=0)
        self.common_keys = most_common.keys()
        self.est = preprocessing.LabelBinarizer()
        self.est.fit(self._transform_uncommon(row))

    def transform(self, data):
        # self.est = preprocessing.KBinsDiscretizer(
        # n_bins = bins, encode = 'onehot', strategy = strategy)
        row = self._transform_na(data)
        row = self._transform_uncommon(row)
        row = self.est.transform(row)
        for i in range(row.shape[1]):
            print(self.k, i)
            data['%s_label_%i' % (self.k, i)] = row[:, i]
        data.drop(self.k, axis=1, inplace=True)
        print('k removed', self.k)

    def _transform_na(self, data):
        return data[self.k].fillna(self.na_value, inplace=False)

    def _transform_uncommon(self, row):
        # Replace all values that are not in `common_keys`
        return util.data.replace_uncommon(row, self.common_keys,
                                          self.uncommon_value)


# class IdCleaner(Estimator):
#     """ Clean numerical fields that represent an id
#     """
#     def fit(self, data):
#         print_primary('\nclean id in `%s`' % self.k)
#         self.est = LabelBinarizer(self.k)
#         self.est.fit(data)

class DeltaStarRating(Estimator):
    k = 'delta_starrating'

    def fit(self, data):
        # TODO standard float-normalization (log?)
        pass

    def transform(self, data):
        data[DeltaStarRating.k] = data['prop_starrating'] - \
            data['visitor_hist_starrating']


# class StarRating(Estimator):
#     def fit(self, data):
#
#         data = data.join(pd.get_dummies(
#             data['prop_starrating'], prefix='prop_starrating_bool'))
#         columns.append('prop_starrating_bool')
#         # data = data.join(pd.get_dummies(
#         # data['visitor_hist_starrating'], prefix='hist_starrating_bool'))

    # def transform(self, data):
    #     p


class PdEstimator:
    def __init__(self, est: TransformerMixin, column_name: str):
        self.est = est
        self.column_name = column_name

    def fit(self, data):
        self.est.fit(data[k])

    def transform(self, data):
        data[k] = self.est.transform(data[k])
