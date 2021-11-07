# -*- coding: utf-8 -*-
# @File    : NewMinMaxScaler.py
# @Author  : Hua Guo
# @Time    : 2021/11/7 下午12:08
# @Disc    :
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, List


class NewMinMaxSaler(MinMaxScaler):
    """
    comparable with null value & numerical input
    """
    def __init__(self, dense_cols: List[str], feature_range=[0, 1]) -> None:
        super(MinMaxScaler, self).__init__()
        self.mms_encoder = MinMaxScaler(
            feature_range=feature_range
        )
        self.dense_cols = dense_cols

    def fit(self, X, y=None):
        self.mms_encoder.fit(X[self.dense_cols])
        return self

    def transform(self, X):
        X.loc[:, self.dense_cols] = self.mms_encoder.transform(X[self.dense_cols])
        return X