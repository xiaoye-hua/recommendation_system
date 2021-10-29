# -*- coding: utf-8 -*-
# @File    : test_NCFFeatureCreator.py
# @Author  : Hua Guo
# @Time    : 2021/10/29 上午9:36
# @Disc    :
from unittest import TestCase
import pandas as pd
import os

from src.FeatureCreator.NCFFeatureCreator import NCFFeatureCreator
from test.config import raw_data_dir, cleaned_data_dir
from src.config import csv_sep


class TestNCFFeatureCreator(TestCase):
    def setUp(self) -> None:
        self.fc = NCFFeatureCreator(profile_feature_dir=raw_data_dir)

    def test_get_feature(self):
        df = pd.read_csv(os.path.join(cleaned_data_dir, 'train.csv'), sep=csv_sep)
        features = self.fc.get_features(uid_item_df=df)
        # print(features.shape)
        self.assertTrue(df.shape[0]==features.shape[0])
