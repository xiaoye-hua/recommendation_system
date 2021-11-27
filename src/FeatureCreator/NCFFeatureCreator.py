# -*- coding: utf-8 -*-
# @File    : NCFFeatureCreator.py
# @Author  : Hua Guo
# @Time    : 2021/10/29 上午9:00
# @Disc    :
import pandas as pd
import os
from typing import Tuple, Dict, Optional, List

from src.FeatureCreator import BaseFeatureCreator
from src.config import origin_data_sep, origin_user_cols, orgin_movie_cols


class NCFFeatureCreator(BaseFeatureCreator):
    def __init__(self, profile_feature_dir: str):
        super(NCFFeatureCreator, self).__init__()
        self.user_profile = self._load_origin_data(file_path=os.path.join(profile_feature_dir, 'users.dat'), cols=origin_user_cols)
        self.item_profile = self._load_origin_data(file_path=os.path.join(profile_feature_dir, 'movies.dat'), cols=orgin_movie_cols)
        self.transformer = None

    def get_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        uid_item_df = df.merge(self.user_profile, how='left', on='user_id').merge(self.item_profile, how='left', on='movie_id')
        return uid_item_df

    def _load_origin_data(self, file_path: str, cols: List[str]) -> pd.DataFrame:
        return pd.read_csv(file_path, sep=origin_data_sep, header=None, names=cols)