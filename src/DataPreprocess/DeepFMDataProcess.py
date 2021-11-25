# -*- coding: utf-8 -*-
# @File    : DeepFMDataProcess.py
# @Author  : Hua Guo
# @Disc    :
from sklearn.base import BaseEstimator, TransformerMixin


class DeepFMDataProcess(BaseEstimator, TransformerMixin):
    def __init__(self):
        super(DeepFMDataProcess, self).__init__()

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        pass