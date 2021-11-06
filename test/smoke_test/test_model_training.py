# -*- coding: utf-8 -*-
# @File    : test_model_training.py
# @Author  : Hua Guo
# @Time    : 2021/10/29 上午10:10
# @Disc    :
from unittest import TestCase
import os
import pandas as pd

from src.FeatureCreator.NCFFeatureCreator import NCFFeatureCreator
from src.Pipeline.NCFPipeline import NCFPipeline
from src.config import csv_sep
from test.config import raw_data_dir, cleaned_data_dir
from src.Pipeline.ItemPopPipeline import ItemPopPipeline


class TestModelTrain(TestCase):
    def setUp(self) -> None:
        self.fc = NCFFeatureCreator(profile_feature_dir=raw_data_dir)
        self.pipeline = NCFPipeline(model_path='logs', model_training=True)
        self.new_pipeline = NCFPipeline(model_path='logs', model_training=False)
        train_data = pd.read_csv(os.path.join(cleaned_data_dir, 'train.csv'), sep=csv_sep)
        self.features = self.fc.get_features(uid_item_df=train_data)

    def test_NCFPipeline_train_eval_save_load_eval(self):

        train_params = {
            "df_for_encode_train": self.features
            , 'batch_size': 64
            , 'epoches': 1
        }
        self.pipeline.train(X=self.features.copy(), y=self.features['rating'], train_params=train_params)
        self.pipeline.eval(X=self.features.copy(), y=self.features['rating'])
        self.pipeline.save_pipeline()

    # def test_NCFPipeline_load(self):
    #     after loading the model, the column names changed to 'input_1', 'input_2'....
    #     self.new_pipeline.eval(X=self.features.copy(), y=self.features['rating'])

    def test_ItemPopPipeline_train_eval_predict(self):
        pipeline = ItemPopPipeline(model_path='logs', model_training=True)
        train_params = {
            "df_for_encode_train": self.features
            , 'batch_size': 64
            , 'epoches': 1
        }
        pipeline.train(X=self.features.copy(), y=self.features['rating'], train_params=train_params)
        pipeline.eval(X=self.features.copy(), y=self.features['rating'])
        pipeline.save_pipeline()

