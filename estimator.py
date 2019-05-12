import pandas as pd
from sklearn import preprocessing
import util.data
from util.string import print_primary, print_secondary, print_warning

# Pipeline of Estimators


def fit_transform(steps, data: pd.DataFrame):
    # steps : iterable of Estimator
    for est in steps:
        print(est)
        print(est.k)
        est.extend(data)
        est.fit(data)
        est.transform(data)


def transform(steps, data: pd.DataFrame):
    # steps : iterable of Estimator
    for est in steps:
        est.extend(data)
        est.transform(data)


# Estimator


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

# Estimator Subclasses


class Wrapper(Estimator):
    """ Wrapper for sklearn.preprocessing estimators
    """

    def __init__(self, k: str, est):
        super().__init__(k)
        self.est = est

    def fit(self, data):
        print('\n', self.est)
        self.est.fit(data[self.k])  # TODO .values or reshape (-1,1)

    def transform(self, data):
        data[self.k] = self.est.transform(data[self.k])


def MinMaxScaler(k):
    return Wrapper(k, preprocessing.MinMaxScaler(feature_range=(-1, 1)))


def RobustScaler(k):
    return Wrapper(k, preprocessing.RobustScaler())


def PowerTransformer(k):
    return Wrapper(k, preprocessing.PowerTransformer())


class Dummy(Estimator):
    # Placeholder, i.e. identity transformation

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
        row = data[self.k].values
        # X = np.array([x for x in X]).reshape(-1, 1)
        # bins = np.repeat(n_bins, X.shape[1])  # e.g. [5,3] for 2 features
        # encode to integers
        # quantile: each bin contains approx. the same number of features
        print(data[self.k].dtype)
        strategy = 'uniform' if util.data.is_int(data[self.k]) else 'quantile'
        # TODO encode='onehot'
        self.est = preprocessing.KBinsDiscretizer(
            n_bins=self.n_bins, encode='onehot', strategy=strategy)
        self.est.fit(row)
        self.n_bins = self.est.bin_edges_[0].size

    def transform(self, data):
        s = ''
        if util.data.is_int(data[self.k]):
            print('\tAttribute & Number of bins (categories)')
            print('\t%s & %i \\\\' % (strategy, n_bins))
        else:
            print('\tbins (%i, %s):' % (n_bins, strategy))
            print('\tAttribute & Bin start & Bin 1 & Bin 2 \\\\')
            for st in [round(a, 3) for a in self.est.bin_edges_[0]]:
                s += '$%s$ & ' % str(st)
            print('\t\t%s & %s\n' % (k, s[:-2]))

        rows = self.est.transform(data[k])

        util.data.join_inplace(data, rows, self.k, k_suffix='bin')
        # data.drop(self.k, axis=1, inplace=True)
        data.drop(columns=self.k, inplace=True)
        print('k removed', self.k)


class LabelBinarizer(Estimator):
    """ Wrapper for sklearn.preprocessing.LabelBinarizer, allowing
    pandas.DataFrame mutations

    Can discretize (onehot-encode) categorical attributes (or ints). The least
    occuring categories are grouped
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
        # self.est = preprocessing.KBinsDiscretizer(
        # n_bins = bins, encode = 'onehot', strategy = strategy)
        row = self._transform_na(data)
        row = self._transform_uncommon(row)
        rows = self.est.transform(row)
        # for i in range(row.shape[1]):
        # print(self.k, i)
        # data['%s_label_%i' % (self.k, i)] = row[:, i]
        # data.drop(self.k, axis=1, inplace=True)
        # print('k removed', self.k)
        util.data.join_inplace(data, rows, self.k)
        # data.drop(self.k, axis=1, inplace=True)
        data.drop(columns=self.k, inplace=True)

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
#
# class DeltaStarRating(Estimator):
#     k = 'delta_starrating'
#
#     def fit(self, data):
#         # TODO standard float-normalization (log?)
#         pass
#
#     def transform(self, data):
#         data[DeltaStarRating.k] = data['prop_starrating'] - \
#             data['visitor_hist_starrating']
#

# class MinMaxScaler(Estimator):
#     """ Wrapper for sklearn.preprocessing.MinMaxScaler
#     """
#
#     def fit(self, data):
#         self.est=preprocessing.MinMaxScaler(feature_range=(-1, 1))
#         self.est.fit(data[self.k])
#
#     def transform(self, data):
#         self.est.transform(data[self.k])
#
#
# class RobustScaler(Estimator):
#     """ Wrapper for sklearn.preprocessing.RobustScaler
#     Normalize based on zero median (instead of zero mean)
#     """
#
#     def fit(self, data):
#         self.est=preprocessing.RobustScaler()
#         self.est.fit(data[self.k])
#
#     def transform(self, data):
#         self.est.transform(data[self.k])
#
# class ZScoreNormalizer(Estimator):
    # assert False


# class LogNormalizer(Estimator):
#     assert False

# class ExtendAttributes(Enum):
