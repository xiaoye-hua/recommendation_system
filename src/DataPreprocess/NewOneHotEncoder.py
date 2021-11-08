# -*- coding: utf-8 -*-
# @File    : NewOneHotEncoder.py
# @Author  : Hua Guo
# @Time    : 2021/11/8 下午8:23
# @Disc    :
import pandas as pd
from typing import Optional, List

from sklearn.preprocessing import OneHotEncoder


class NewOneHotEncoder(OneHotEncoder):
    """
    comparable with null value & numerical input
    """
    def __init__(self, sparse_cols: List[str]) -> None:
        super(NewOneHotEncoder, self).__init__()
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.sparse_cols = sparse_cols

    def fit(self, X, y=None):
        self.one_hot_encoder.fit(X[self.sparse_cols])
        return self

    def transform(self, X):
        test_df = pd.DataFrame(self.one_hot_encoder.transform(X[self.sparse_cols]).toarray())
        X = X.drop(self.sparse_cols, axis=1)
        X = X.reset_index().drop('index', axis=1).merge(test_df, how='left', left_index=True, right_index=True)
        return X