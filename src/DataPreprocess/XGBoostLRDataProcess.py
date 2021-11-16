# -*- coding: utf-8 -*-
# @File    : XGBoostLRDataProcess.py
# @Author  : Hua Guo
# @Disc    :
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from xgboost.sklearn import XGBModel
from copy import deepcopy
from xgboost.sklearn import XGBClassifier
import logging
logging.getLogger(__name__)


class XGBoostLRDataProcess(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        self.xgb = XGBClassifier()
        self.one_hot = OneHotEncoder()

    def fit(self, X, y):
        X = deepcopy(X)
        self.xgb.fit(
            X=X, y=y
            , verbose=True
            , eval_metric='logloss'

            # , verbose=self.xgb_train_params['verbose']
            # , eval_metric=self.xgb_train_params['eval_metric']
            ,eval_set=[[X, y]]
                     )

        X = self.xgb.apply(X)
        self.one_hot.fit(X)
        return self

    def transform(self, X, y=None):
        X = self.xgb.apply(X)  # [:, 70:]
        X = self.one_hot.transform(X)
        return X
