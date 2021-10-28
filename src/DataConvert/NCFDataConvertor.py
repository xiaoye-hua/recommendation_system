# -*- coding: utf-8 -*-
# @File    : NCFDataConvertor.py
# @Author  : Hua Guo
# @Time    : 2021/10/28 下午10:42
# @Disc    : Data convertor for neural collaborative filtering
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

from src.DataConvert import BaseDataConvertor, get_train_test_data, split_data_ml100k, read_data_ml100k
from src.config import csv_sep, origin_rating_cols, orgin_movie_cols, origin_user_cols


class NCFDataConvertor(BaseDataConvertor):
    """
    output:
        1. train data
        2. eval data
            1. ground truth data
            2. candidate data for prediction
        Raw features are stored in the above dataset
    """
    def __init__(self, input_dir: str, output_dir: str, split_mode='random', test_ratio=0.1, negsample=3) -> None:
        """

        :param input_dir:
        :param output_dir:
        :param split_mode:  'random' or 'seq-aware'
        :param test_ratio:
        :param negsample:
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self._check_dir(self.input_dir)
        self._check_dir(self.output_dir)
        self.negsample = negsample
        self.split_mode = split_mode
        self.test_ratio = test_ratio
        self.user_profile, self.item_profile, self.positive_train, self.group_truth = self._load_data()

    def _load_data(self):
        origin_sep = "::"
        user_profile = pd.read_csv(os.path.join(self.input_dir, 'users.dat'), sep=origin_sep, header=None, names=origin_user_cols)
        item_profile = pd.read_csv(os.path.join(self.input_dir, 'movies.dat'), sep=origin_sep, header=None, names=orgin_movie_cols)
        # item_profile['movie_id'] = item_profile['movie_id'].astype('int64')
        data, num_users, num_items = read_data_ml100k(raw_data_dir=self.input_dir)
        positive_train, test_data = split_data_ml100k(data, num_users,
                                                  self.split_mode, self.test_ratio
                                                      )
        return user_profile, item_profile, positive_train, test_data

    def _get_neg_data(self):
        assert self.candidate_data is not None
        self.negative_train = self.candidate_data.groupby('user_id').sample(n=self.negsample, replace=True)

    def _get_candidate_data(self):
        self.user_profile = self.user_profile.assign(
            key=0
        )
        self.item_profile = self.item_profile.assign(
            key=0
        )
        all_combination = self.user_profile.merge(self.item_profile, how='left', on='key')
        all_combination = all_combination.merge(self.positive_train, how='left', on=['user_id', 'movie_id'])
        self.candidate_data = all_combination[all_combination['rating'].isna()][['user_id', 'movie_id']]

    def _save_data(self) -> None:
        self.positive_train = self.positive_train[['user_id', 'movie_id']]
        self.group_truth = self.group_truth[['user_id', 'movie_id', 'rating']]
        self.negative_train = self.negative_train.assign(
            rating=0
        )
        self.positive_train = self.positive_train.assign(
            rating=1
        )
        self.train = pd.concat([self.negative_train, self.positive_train], axis=0)
        self._save_csv(df=self.train, file_name=os.path.join(self.output_dir, 'train.csv'))
        self._save_csv(df=self.candidate_data, file_name=os.path.join(self.output_dir, 'candidate.csv'))
        self._save_csv(df=self.group_truth, file_name=os.path.join(self.output_dir, 'test.csv'))

    def _save_csv(self, df: pd.DataFrame, file_name: str) -> None:
        print(f"Data dir: {file_name}; data shape: {df.shape}")
        df.to_csv(file_name, index=False, sep=csv_sep)

    def convert(self):
        self._get_candidate_data()
        self._get_neg_data()
        self._save_data()