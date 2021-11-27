# -*- coding: utf-8 -*-
# @File    : NCFPipeline.py
# @Author  : Hua Guo
# @Time    : 2021/10/29 上午9:21
# @Disc    :
import pandas as pd
import os
import tensorflow as tf
from typing import Tuple
import joblib
import logging
from deepmatch.models import NCF
from sklearn.pipeline import Pipeline

from src.Pipeline import BasePipeline
from src.DataPreprocess.NewOrdinalEncoder import NewOrdinalEncoder
from src.utils.plot_utils import binary_classification_eval
logging.getLogger(__name__)


class NCFPipeline(BasePipeline):
    def __init__(self, model_path: str, model_training=False, **kwargs):
        super(NCFPipeline, self).__init__(model_path=model_path, model_training=model_training)
        self.user_features = ['user_id', 'gender', 'age', 'occupation', 'zip']
        self.item_features = ['movie_id']
        self.all_features = self.user_features + self.item_features
        self.eval_result_path = os.path.join(self.model_path, 'eval')
        self.model_file_name = 'model.pb'
        self.preprocess_file_name = 'preprocess.pkl'
        if model_training:
            self.preprocess_pipeline = None
            self.model = None
        else:
            self.preprocess_pipeline, self.model = self.load_pipeline()

    def _process_train_data(self, X):
        train_model_input = {}
        if self.model_training:
            for col in self.all_features:
                train_model_input[col] = X[col].values
        else:
            for idx, col in enumerate(self.all_features):
                target_col = f"input_{idx+1}"
                train_model_input[target_col] = X[col].values
        return train_model_input

    def train(self, X, y, train_params):
        pre_process_pipeline_lst = []
        df_for_encode_train = train_params['df_for_encode_train']
        batch_size = train_params['batch_size']
        epoches = train_params['epoches']

        # train_valid = train_params["train_valid"]
        transformer = NewOrdinalEncoder(category_cols=self.all_features, begin_idx=1)
        transformer.fit(df_for_encode_train)
        X = transformer.transform(X=X)
        pre_process_pipeline_lst.append(
            ('ordinal_transformer', transformer)
        )
        self.preprocess_pipeline = Pipeline(pre_process_pipeline_lst)

        train_model_input = self._process_train_data(X=X)

        train_label = y.values
        user_feature_columns = {}
        item_feature_columns = {}
        for target_dic, cols in zip([user_feature_columns, item_feature_columns], [self.user_features, self.item_features]):
            for col in cols:
                target_dic[col] = X[col].max() + 1
        self.model = NCF(user_feature_columns, item_feature_columns, user_gmf_embedding_dim=20,
                    item_gmf_embedding_dim=20, user_mlp_embedding_dim=32, item_mlp_embedding_dim=32,
                    dnn_hidden_units=[128, 64, 32], )
        self.model.summary()
        self.model.compile("adam", "binary_crossentropy",
                      metrics=['binary_crossentropy', tf.keras.metrics.AUC()], )

        self.model.fit(train_model_input, train_label,
                        batch_size=batch_size,
                       epochs=epoches,
                       # verbose=2,
                       validation_split=0.2,
                       )

    def predict_proba(self, X):
        X = self.preprocess_pipeline.transform(X)
        X = self._process_train_data(X=X)
        return self.model.predict(X)

    def eval(self, X: pd.DataFrame, y: pd.DataFrame, default_fig_dir='logs') -> None:
        if default_fig_dir is None:
            fig_dir = self.eval_result_path
        else:
            fig_dir = default_fig_dir
        self._check_dir(fig_dir)
        # X = self.preprocess_pipeline.transform(X=X.copy())
        # X = self._process_train_data(X=X)
        # predict_prob = self.model.predict(X)
        predict_prob = self.predict_proba(X)
        binary_classification_eval(test_y=y, predict_prob=predict_prob, fig_dir=fig_dir)

    def load_pipeline(self) -> Tuple[Pipeline, tf.keras.models.Model]:
        logging.info(f"Loading model from {self.model_path}...")
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