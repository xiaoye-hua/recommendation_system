# -*- coding: utf-8 -*-
# @File    : test_model_train_eval_save_load_inference.py
# @Author  : Hua Guo
# @Time    : 2021/10/29 上午10:10
# @Disc    :
from unittest import TestCase
import os
import pandas as pd

from src.FeatureCreator.NCFFeatureCreator import NCFFeatureCreator
from src.FeatureCreator.DeepFMFeatureCreator import DeepFMFeatureCreator
from src.Pipeline.NCFPipeline import NCFPipeline
from src.config import csv_sep, criteo_target_col, criteo_csv_sep, criteo_df_cols
from test.config import raw_data_dir, cleaned_data_dir, criteo_data_dir
from src.Pipeline.ItemPopPipeline import ItemPopPipeline
from src.Pipeline.DeepFMPipeline import DeepFMPipeline
from src.Pipeline.WideDeepPipeline import WideDeepPipeline
from src.Pipeline.LogisticRegressionPipeline import LogisticRegressionPipeline


class TestModelTrain(TestCase):
    def setUp(self) -> None:
        self.fc = NCFFeatureCreator(profile_feature_dir=raw_data_dir)
        self.pipeline = NCFPipeline(model_path='logs', model_training=True)
        # self.new_pipeline = NCFPipeline(model_path='logs', model_training=False)
        train_data = pd.read_csv(os.path.join(cleaned_data_dir, 'train.csv'), sep=csv_sep)
        self.features = self.fc.get_features(df=train_data)

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

    def test_DeepFM_train_eval_save_load_eval(self):
        model_dir = "logs/DeepFM_train_eval_save_load_eval/"
        # features = pd.read_csv(criteo_data_dir)
        features = pd.read_csv(criteo_data_dir,
                                 sep=criteo_csv_sep,
                                 header=None, names=criteo_df_cols
                               , nrows=10
                               )
        features, feature_cols = DeepFMFeatureCreator().get_features(features)
        train_params = {
            "df_for_encode_train": features.copy()[feature_cols]
            , 'batch_size': 64
            , 'epoches': 1
            , 'dense_to_sparse': True
        }
        DeepFM = DeepFMPipeline(model_path=model_dir, model_training=True)
        DeepFM.train(X=features.copy()[feature_cols], y=features[criteo_target_col], train_params=train_params)
        DeepFM.eval(X=features.copy()[feature_cols], y=features[criteo_target_col])
        DeepFM.save_pipeline()
        new_DeepFM = DeepFMPipeline(model_path=model_dir, model_training=False)
        new_DeepFM.eval(X=features.copy()[feature_cols], y=features[criteo_target_col])

    def test_WDL_train_eval_save_load_eval(self):
        # features = pd.read_csv(criteo_data_dir)
        features = pd.read_csv(criteo_data_dir,
                                 sep=criteo_csv_sep,
                                 header=None, names=criteo_df_cols
                               , nrows=10
                               )
        features = DeepFMFeatureCreator().get_features(features)
        train_params = {
            "df_for_encode_train": features.copy()
            , 'batch_size': 64
            , 'epoches': 1
        }
        wide_deep = WideDeepPipeline(model_path='logs', model_training=True)
        wide_deep.train(X=features.copy(), y=features[criteo_target_col], train_params=train_params)
        wide_deep.predict_proba(X=features.copy())
        # DeepFM.eval(X=features.copy(), y=features[criteo_target_col])
        # DeepFM.save_pipeline()

    def test_LogisticRegression_train_eval_save_load_eval(self):
        features = pd.read_csv(criteo_data_dir,
                                 sep=criteo_csv_sep,
                                 header=None, names=criteo_df_cols
                               , nrows=100
                               )
        features, feature_cols  = DeepFMFeatureCreator().get_features(features)
        train_params = {
            "df_for_encode_train": features.copy()
            , 'batch_size': 64
            , 'epoches': 1
        }
        pipeline = LogisticRegressionPipeline(model_path='logs/test1114', model_training=True)
        pipeline.train(X=features.copy()[feature_cols], y=features[criteo_target_col], train_params=train_params)
        pipeline.eval(X=features.copy()[feature_cols], y=features[criteo_target_col])
        pipeline.save_pipeline()
        new_pipeline = LogisticRegressionPipeline(model_path='logs/test1114', model_training=False)
        new_pipeline.eval(X=features.copy()[feature_cols], y=features[criteo_target_col])

