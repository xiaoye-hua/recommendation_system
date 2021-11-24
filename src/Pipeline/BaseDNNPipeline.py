# -*- coding: utf-8 -*-
# @File    : BaseDNNPipeline.py
# @Author  : Hua Guo
# @Disc    :
import pandas as pd
import os
import tensorflow as tf
from typing import Tuple
import joblib
import logging
from sklearn.pipeline import Pipeline

from src.Pipeline import BasePipeline
logging.getLogger(__name__)


class BaseDNNPipeline(BasePipeline):
    def __init__(self, model_path: str, model_training=False, **kwargs):
        super(BaseDNNPipeline, self).__init__(model_path=model_path, model_training=model_training)
        self.model_file_name = 'model.pb'
        self.preprocess_file_name = 'preprocess.pkl'
        if model_training:
            self.preprocess_pipeline = None
            self.model = None
        else:
            self.preprocess_pipeline, self.model = self.load_pipeline()

    def predict_proba(self, X):
        X = self.preprocess_pipeline.transform(X)
        return self.model.predict(X)

    def load_pipeline(self) -> Tuple[Pipeline, tf.keras.models.Model]:
        pipeline = tf.keras.models.load_model(self.model_path)
        pre_pipeline = joblib.load(
            filename=os.path.join(self.model_path, self.preprocess_file_name)
        )
        return pre_pipeline, pipeline

    def save_pipeline(self) -> None:
        file_name = joblib.dump(
            value=self.preprocess_pipeline,
            filename=os.path.join(self.model_path, self.preprocess_file_name)
        )[0]
        tf.keras.models.save_model(model=self.model, filepath=self.model_path)
        logging.info(f'Model saved in {self.model_path}')

    def train(self, X, y, train_params):
        pass