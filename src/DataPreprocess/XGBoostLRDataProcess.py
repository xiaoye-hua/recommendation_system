# -*- coding: utf-8 -*-
# @File    : XGBoostLRDataProcess.py
# @Author  : Hua Guo
# @Disc    :
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from xgboost.sklearn import XGBModel
from copy import deepcopy
from xgboost.sklearn import XGBClassifier
import logging
logging.getLogger(__name__)


class XGBoostLRDataProcess(OrdinalEncoder):
    def __init__(self, # xgb_model: XGBModel #, train_params: dict
                    ) -> None:
        super(XGBoostLRDataProcess, self).__init__()
        self.xgb = XGBClassifier()
        # self.xgb_train_params = train_params
        self.one_hot = OneHotEncoder()

    def fit(self, X, y=None):
        X = deepcopy(X)
        self.xgb.fit(
            X=X, y=y
            , verbose = True, eval_metric = 'logloss'
            # , verbose=self.xgb_train_params['verbose']
            # , eval_metric=self.xgb_train_params['eval_metric']
            # ,eval_set=self.xgb_train_params['eval_set']
                     )

        X = self.xgb.apply(X)
        self.one_hot.fit(X)
        return self

    def transform(self, X):
        X = self.xgb.apply(X)  # [:, 70:]
        X = self.one_hot.transform(X)
        return X
