# -*- coding: utf-8 -*-
# @File    : XGBoostLRDataProcess.py
# @Author  : Hua Guo
# @Disc    :
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from xgboost.sklearn import XGBModel
from copy import deepcopy


class XGBoostLRDataProcess(OrdinalEncoder):
    def __init__(self, xgb_model: XGBModel, train_params: dict) -> None:
        super(XGBoostLRDataProcess, self).__init__()
        self.xgb = xgb_model
        self.xgb_train_params = train_params
        self.one_hot = OneHotEncoder()

    def fit(self, X, y=None, train_params={}):
        X = deepcopy(X)
        self.xgb_train_params.update(
            {
                'X': X
                , 'y': y
            }
        )
        self.xgb.fit(
            # X=X, y=y
            #          , verbose=True, eval_metric='logloss'
            #          , eval_set=[[train_X, train_y], [test_X, test_y]]
            **self.xgb_train_params
                     )
        X = self.xgb.apply(X)
        self.one_hot.fit(X)

    def transform(self, X):
        X = self.xgb.apply(X)  # [:, 70:]
        X = self.one_hot.transform(X)
        return X
