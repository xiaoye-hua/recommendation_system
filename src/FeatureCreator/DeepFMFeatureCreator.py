# -*- coding: utf-8 -*-
# @File    : DeepFMFeatureCreator.py
# @Author  : Hua Guo
# @Time    : 2021/11/7 上午9:40
# @Disc    :
import pandas as pd

from src.FeatureCreator import BaseFeatureCreator
from src.config import criteo_dense_features, criteo_sparse_features


class DeepFMFeatureCreator(BaseFeatureCreator):

    def get_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df[criteo_sparse_features] = df[criteo_sparse_features].fillna('-1', )
        df[criteo_dense_features] = df[criteo_dense_features].fillna(0, )
        return df
