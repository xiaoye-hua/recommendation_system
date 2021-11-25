# -*- coding: utf-8 -*-
# @File    : DeepFMPipeline.py
# @Author  : Hua Guo
# @Time    : 2021/11/6 下午7:55
# @Disc    :
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, KBinsDiscretizer
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat
from sklearn.compose import ColumnTransformer

from src.config import criteo_sparse_features, criteo_dense_features
from src.Pipeline.BaseDNNPipeline import BaseDNNPipeline
from src.DataPreprocess.DeepFMDataProcess import DeepFMDataProcess
logging.getLogger(__name__)


class DeepFMPipeline(BaseDNNPipeline):
    def __init__(self, model_path: str, model_training=False, **kwargs):
        super(DeepFMPipeline, self).__init__(model_path=model_path, model_training=model_training, **kwargs)
        self.model_file_name = 'model.pb'
        self.preprocess_file_name = 'preprocess.pkl'
        self.model_training = model_training
        if self.model_training:
            self.preprocess_pipeline = None
            self.model = None
        else:
            self.preprocess_pipeline, self.model = self.load_pipeline()

    def train(self, X, y, train_params):
        df_for_encode_train = train_params['df_for_encode_train']
        batch_size = train_params['batch_size']
        epoches = train_params['epoches']
        dense_to_sparse = train_params['dense_to_sparse']
        self.preprocess_pipeline = DeepFMDataProcess(dense_feature=criteo_dense_features, sparse_feature=criteo_sparse_features
                                                     ,dense_to_sparse=dense_to_sparse)
        logging.info(self.preprocess_pipeline)

        df_for_encode_train = self.preprocess_pipeline.fit_transform(df_for_encode_train)
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=df_for_encode_train[feat].max() + 1, embedding_dim=4)
                                  for i, feat in enumerate(criteo_sparse_features)] + [DenseFeat(feat, 1, )
                                                                                for feat in criteo_dense_features]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns
        X = self.preprocess_pipeline.transform(X)
        train_model_input = self._process_train_data(X)
        train_label = y.values
        self.model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
        self.model.summary()
        self.model.compile("adam", "binary_crossentropy",
                      metrics=['binary_crossentropy', tf.keras.metrics.AUC()], )

        self.model.fit(train_model_input, train_label,
                        batch_size=batch_size,
                       epochs=epoches,
                       # verbose=2,
                       validation_split=0.2,
                       )
        print()

    def _process_train_data(self, X):
        train_model_input = {}
        if self.model_training:
            for col in criteo_sparse_features+criteo_dense_features:
                train_model_input[col] = X[col]
        else:
            for idx, col in enumerate(criteo_sparse_features+criteo_dense_features):
                target_col = f'input_{idx+1}'
                train_model_input[target_col] = X[col]
        return train_model_input

    def predict_proba(self, X):
        X = self.preprocess_pipeline.transform(X)
        trian_input = self._process_train_data(X)
        prob = self.model.predict(trian_input)
        return prob