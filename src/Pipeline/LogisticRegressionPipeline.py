# -*- coding: utf-8 -*-
# @File    : LogisticRegressionPipeline.py
# @Author  : Hua Guo
# @Time    : 2021/11/8 下午7:04
# @Disc    :
import joblib
import os
import logging
logging.getLogger(__name__)

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.Pipeline import BasePipeline
from src.DataPreprocess.NewOrdinalEncoder import NewOrdinalEncoder
from src.DataPreprocess.NewOneHotEncoder import NewOneHotEncoder
from src.config import criteo_sparse_features
from src.utils.plot_utils import plot_feature_importances, binary_classification_eval


class LogisticRegressionPipeline(BasePipeline):
    def __init__(self, model_path: str, model_training=False, **kwargs):
        super(LogisticRegressionPipeline, self).__init__(model_path=model_path, model_training=model_training)
        self.pipeline = None
        self.model_file_name = "model.pkl"
        self.eval_result_path = os.path.join(self.model_path, 'eval')

    def train(self, X, y, train_params):
        pipeline_lst = []
        df_for_encode_train = train_params['df_for_encode_train']
        # train_valid = train_params["train_valid"]
        one_hot = train_params.get('one_hot', False)
        if one_hot:
            transformer = NewOneHotEncoder(sparse_cols=criteo_sparse_features)
        else:
            transformer = NewOrdinalEncoder(category_cols=criteo_sparse_features)
        transformer.fit(df_for_encode_train)
        logging.info(f"Data dims before transformer: {X.shape}")
        X = transformer.transform(X=X)
        logging.info(f"Data dims after transformer: {X.shape}")
        pipeline_lst.append(("new_ordinal_transformer", transformer))
        if train_params.get('pca_component_num', False):
            pca_component_num = train_params['pca_component_num']
            # sc_V = MinMaxScaler()
            pca = PCA(pca_component_num)
            pipeline_lst.extend([
                # ('min_max', sc_V)
                # ,
                ('pca', pca)
            ])
            # X = sc_V.fit_transform(X)
            X = pca.fit_transform(X)

        model = LogisticRegression(
            penalty='l1', solver='saga'
        )
        model.fit(X=X, y=y)
        pipeline_lst.append(('lr', model))
        self.pipeline = Pipeline(pipeline_lst)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X=X)[:, 1]

    def save_pipeline(self):
        file_name = joblib.dump(
            value=self.pipeline,
            filename=os.path.join(self.model_path, self.model_file_name)
        )[0]

    def load_pipeline(self):
        self.pipeline = joblib.load(
            filename=os.path.join(self.model_path, self.model_file_name)
        )
