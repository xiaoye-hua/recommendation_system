# -*- coding: utf-8 -*-
# @File    : NCFFeatureCreator.py
# @Author  : Hua Guo
# @Time    : 2021/10/29 上午9:00
# @Disc    :
import pandas as pd
import numpy as np
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

    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        uid_item_df = df.merge(self.user_profile, how='left', on='user_id').merge(self.item_profile, how='left', on='movie_id')
        return uid_item_df

    def _load_origin_data(self, file_path: str, cols: List[str]) -> pd.DataFrame:
        return pd.read_csv(file_path, sep=origin_data_sep, header=None, names=cols)

# def gen_model_input_new(train_set: pd.DataFrame, user_profile, item_profile, seq_max_len, user_features,
#                         item_features, user_col, item_col, label_col=None):
#
#     # train_uid = np.array([line[0] for line in train_set])
#     # train_seq = [line[1] for line in train_set]
#     # train_iid = np.array([line[2] for line in train_set])
#     # train_label = np.array([line[3] for line in train_set])
#     # train_hist_len = np.array([line[4] for line in train_set])
#     train_label = None
#     train_uid = train_set[user_col].values
#     train_iid = train_set[item_col].values
#     if label_col is not None:
#         train_label = train_set[label_col].values
#
#     # train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
#     train_model_input = {user_col: train_uid, item_col: train_iid,
#                          # "hist_movie_id": train_seq_pad,
#                          # "hist_len": train_hist_len
#                          }
#
#     for key in list(set(user_features) - set([user_col])):
#         train_model_input[key] = user_profile.loc[train_model_input[user_col]][key].values
#     for key in list(set(item_features) - set([item_col])):
#         train_model_input[key] = item_profile.loc[train_model_input[item_col]][key].values
#
#     return train_model_input, train_label