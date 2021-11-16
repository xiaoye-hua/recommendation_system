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
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
import mlflow
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
from src.Pipeline import BasePipeline
from src.DataPreprocess.NewOrdinalEncoder import NewOrdinalEncoder
from src.DataPreprocess.NewOneHotEncoder import NewOneHotEncoder
from src.config import criteo_sparse_features, criteo_dense_features
from src.utils.plot_utils import plot_feature_importances, binary_classification_eval


class LogisticRegressionPipeline(BasePipeline):
    def __init__(self, model_path: str, model_training=False, **kwargs):
        super(LogisticRegressionPipeline, self).__init__(model_path=model_path, model_training=model_training)
        self.model_file_name = "model.pkl"
        self.eval_result_path = os.path.join(self.model_path, 'eval')
        self.pipeline = None
        if not model_training:
            self.load_pipeline()

    def train(self, X, y, train_params):
        pipeline_lst = []
        df_for_encode_train = train_params['df_for_encode_train']
        # train_valid = train_params["train_valid"]
        one_hot = train_params.get('one_hot', False)
        dense_bin = train_params.get('dense_bin', False)
        dense_standard = train_params.get('dense_standard', False)

        numeric_features = criteo_dense_features  # ["age", "fare"]
        numeric_steps = []
        if dense_standard:
            numeric_steps.append(
                ('standard', StandardScaler())
            )
        if dense_bin:
            numeric_steps.append(
                ('dense_bin', KBinsDiscretizer(n_bins=20, encode='ordinal'))
            )
        numeric_transformer = Pipeline(
            steps=numeric_steps
        )
        categorical_features = criteo_sparse_features  # ["embarked", "sex", "pclass"]
        if one_hot:
            cate_encoder = OneHotEncoder()
            categorical_transformer = Pipeline([
                ('one_hot', cate_encoder)
                , ('pca', TruncatedSVD(n_components=50))
            ])
        else:
            categorical_transformer = OrdinalEncoder()

        transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
        # transformer = make_column_transformer(
        #     (categorical_transformer, make_column_selector(dtype_include="object"))
        #     , remainder='passthrough'
        # )
        transformer.fit(df_for_encode_train)
        logging.info(f"Data transformation steps: ")
        logging.info(transformer)
        logging.info(f"Data dims before transformer: {X.shape}")
        X = transformer.transform(X=X)
        logging.info(f"Data dims after transformer: {X.shape}")
        pipeline_lst.append(("new_ordinal_transformer", transformer))
        model = LogisticRegression(
            penalty='l1', solver='saga', verbose=1
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
        # mlflow.sklearn.save_model(sk_model=self.pipeline
        #                           ,path=self.model_path)

    def load_pipeline(self):
        self.pipeline = joblib.load(
            filename=os.path.join(self.model_path, self.model_file_name)
        )
        # self.pipeline = mlflow.sklearn.load_model(model_uri=self.model_path)
