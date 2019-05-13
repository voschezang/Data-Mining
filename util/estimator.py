import pandas as pd
from sklearn import preprocessing
from sklearn import impute
import util.data
from util.string import print_primary, print_warning


class Estimator:
    # Interface, partly compatible with sklearn.base.TransformerMixin
    def __init__(self, k: str = ''):
        self.k = k
        self.est = None

    def extend(self, data: pd.DataFrame):
        """ Extend the dataframe by adding additional attributes
        This function should be applied to a dataframe before fitting or before
        transforming it
        """
        pass

    def fit(self, data: pd.DataFrame):
        # fit a model and save the parameters
        pass

    def transform(self, data):
        # use the fitted model to predict attribute values
        raise NotImplementedError


###############################################################################
# Estimator Subclasses
###############################################################################

class Wrapper(Estimator):
    """ Wrapper for sklearn.preprocessing estimators
    """

    def __init__(self, k: str, est):
        super().__init__(k)
        self.est = est

    def fit(self, data):
        self.est.fit(data[self.k].values.reshape(-1, 1))

    def transform(self, data):
        if util.data.is_int(data[self.k]):
            data[self.k] = self.est.transform(
                data[self.k].values.reshape(-1, 1)).astype(int)
        else:
            data[self.k] = self.est.transform(
                data[self.k].values.reshape(-1, 1))


def Imputer(k):
    return Wrapper(k, impute.SimpleImputer(strategy='median', copy=False))


def MinMaxScaler(k):
    return Wrapper(k, preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False))


def RobustScaler(k):
    return Wrapper(k, preprocessing.RobustScaler(copy=False))


def PowerTransformer(k):
    return Wrapper(k, preprocessing.PowerTransformer(), copy=False)


class NoTransformEstimator(Estimator):
    # Placeholder, i.e. identity transformation

    def transform(self, data):
        pass


# class Imputer(Estimator):
#     """ Imputer class for pandas.DataFrame
#     """
#
#     def fit(self, data):
#         self.value = data[self.k].median()
#
#     def transform(self, data):
#         # Replace missing values by median
#         data[self.k].fillna(self.value, inplace=True)
#

class Discretizer(Estimator):
    # Bin numerical data
    """ Encode data[k] to a numerical format (in range [0,n_bins])
    Use stragegy=`uniform` when encoding integers (e.g. id's)

    :E Encoder object with attributes `encoders`, `decoders`
    """

    def __init__(self, k):
        super().__init__(k)
        self.n_bins = 5

    def fit(self, data):
        print_primary('\tdicretize `%s`' % self.k)
        # row = data[self.k].values
        # X = np.array([x for x in X]).reshape(-1, 1)
        # bins = np.repeat(n_bins, X.shape[1])  # e.g. [5,3] for 2 features
        # encode to integers
        # quantile: each bin contains approx. the same number of features
        strategy = 'uniform' if util.data.is_int(
            data[self.k]) else 'quantile'
        # TODO encode='onehot'
        self.est = preprocessing.KBinsDiscretizer(
            n_bins=self.n_bins, encode='onehot', strategy=strategy)
        self.est.fit(data[self.k].values.reshape(-1, 1))
        self.n_bins = self.est.bin_edges_[0].size

    def transform(self, data):
        if util.data.is_int(data[self.k]):
            print('\tAttribute & Number of bins (categories)')
            print('\t%s & %i \\\\' % (self.est.strategy, self.est.n_bins))
        else:
            print('\t%s & %i \\\\' % (self.est.strategy, self.est.n_bins))
            print('\tAttribute & Bin start & Bin 1 & Bin 2 \\\\')
            s = ''
            for st in [round(a, 3) for a in self.est.bin_edges_[0]]:
                s += '$%s$ & ' % str(st)
            print('\t\t%s & %s\n' % (self.k, s[:-2]))

        # note the sparse output
        rows = self.est.transform(data[self.k].values.reshape(-1, 1))
        util.data.join_inplace(data, rows, self.k, k_suffix='bin')


class RemoveKey(Estimator):
    # Remove k, e.g. after discretization
    def transform(self, data):
        # print(data[self.k + '_bin0'])
        # data.drop(self.k, axis=1, inplace=True)
        data.drop(columns=self.k, inplace=True)
        print('k removed', self.k)


class LabelBinarizer(Estimator):
    """ Wrapper for sklearn.preprocessing.LabelBinarizer, allowing
    pandas.DataFrame mutations

    Can discretize (onehot-encode) categorical attributes (or ints). The least
    occuring categories are grouped
    """

    def __init__(self, k, use_keys=False):
        # :use_keys = use keys as labels
        super().__init__(k)
        self.n_max = 10  # TODO use larger number
        self.use_keys = use_keys

    def fit(self, data):
        """ Find most common catorgies, group uncommon categories under a
        single `uncommon_value` and fit estimator
        """
        print_primary('\nOneHotEncode labels `%s`' % self.k)
        # max_prop_null_values = 0.05
        # if util.data.proportion_null_values(data, self.k) > max_prop_null_values :
        # flag_null_values(data, self.k)

        # isinstance is not supported by pd.Series.dtype
        if util.data.is_int(data[self.k]):
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
        print('transform')
        # self.est = preprocessing.KBinsDiscretizer(
        # n_bins = bins, encode = 'onehot', strategy = strategy)
        row = self._transform_na(data)
        row = self._transform_uncommon(row)
        rows = self.est.transform(row)
        if self.use_keys:
            util.data.join_inplace(data, rows, self.k, keys=self.est.classes_)
        else:
            util.data.join_inplace(data, rows, self.k)

    def _transform_na(self, data):
        return data[self.k].fillna(self.na_value, inplace=False)

    def _transform_uncommon(self, row):
        # Replace all values that are not in `common_keys`
        return util.data.replace_uncommon(row, self.common_keys,
                                          self.uncommon_value)


class GrossBooking(Estimator):
    def fit(self, data):
        regData = data.loc[~data['gross_bookings_usd'].isnull(), :]
        cols = regData.columns
        keys1 = [k for k in cols if 'bool' in str(k)]
        keys2 = [k for k in cols if 'null' in str(k)]
        keys3 = [k for k in cols if 'able_comp'in str(k)]
        keys4 = [k for k in cols if 'location_score' in str(k)]
        keys5 = [k for k in cols if 'prop_log' in str(k)]
        self.fullK = keys1 + keys2 + keys3 + keys4 + keys5 + ['avg_price_comp']
        self.fullK.remove('booking_bool')
        self.fullK.remove('click_bool')
        self.fullK = [k for k in self.fullK if 'log' not in str(k)]
        self.est = util.data.regress_booking(regData, self.fullK)

    def transform(self, data):
        if 'gross_bookings_usd' in data.columns:
            data.loc[data['gross_bookings_usd'].isnull(), 'gross_bookings_usd'] = \
                self.est.predict(
                    data.loc[data['gross_bookings_usd'].isnull(), self.fullK])
        else:
            # unlabelled data
            data['gross_bookings_usd'] = self.est.predict(data[self.fullK])
