""" A data preprocesing pipeline of estimators. Does not classify data (unlike
sklearn Pipeline) but merely fits estimator models and transforms data.
"""
import pandas as pd
import sys


class Pipeline:
    def __init__(self, steps, data: pd.DataFrame):
        """ Init estimators by fitting them to `data` and use fitted estimators
        to transform `data`
        :steps : iterable of Estimator
        """
        self.steps = steps
        self._fit_transform(data)

    def transform(self, data: pd.DataFrame):
        # Pipeline.transform(self.steps, data)
        """" Transform (unseen) data
        """
        # assert all([k in data.columns for k in self.columns_fitted_data])
        for est in self.steps:
            est.extend(data)
            est.transform(data)
            # print('mem size', sys.getsizeof(data))

    def _fit_transform(self,  data: pd.DataFrame):
        """" Fit and Transfrom estimator models
        """
        for est in self.steps:
            print(est, est.k)
            est.extend(data)
            est.fit(data)
            est.transform(data)
            # print('mem size', sys.getsizeof(data))
