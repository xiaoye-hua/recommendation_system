# -*- coding: utf-8 -*-
# @File    : XGBoostLRPipeline2.py
# @Author  : Hua Guo
# @Disc    :
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from scipy import sparse

import os
import logging
logging.getLogger(__name__)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer

from src.Pipeline import BasePipeline
from src.utils.plot_utils import plot_feature_importances, binary_classification_eval
from src.DataPreprocess.NewOrdinalEncoder import NewOrdinalEncoder
from src.Pipeline.XGBoostPipeline import XGBoostPipeline
from src.config import criteo_dense_features, criteo_sparse_features

from src.DataPreprocess.XGBoostLRDataProcess import XGBoostLRDataProcess


class XGBoostLRPipeline2(XGBoostPipeline):
    def train(self, X: pd.DataFrame, y: pd.DataFrame, train_params: dict) -> None:
        pipeline_lst = []

        df_for_encode_train = train_params['df_for_encode_train']
        train_valid = train_params.get("train_valid", False)
        # transformer = NewOrdinalEncoder(category_cols=self.cate_encode_cols)
        # transformer.fit(df_for_encode_train)
        # X = transformer.transform(X=X)
        # pipeline_lst.append(("new_ordinal_transformer", transformer))
        # self.ordianl = transformer
        self.xgb = XGBClassifier(**self.model_params)
        print(f"Model params are {self.xgb.get_params()}")
        lr_dense_transformer = Pipeline([
            ('standard', StandardScaler())
            ,('dense_bin', KBinsDiscretizer(n_bins=20, encode='ordinal'))
        ])
        cate_encoder = OneHotEncoder()
        lr_sparse_transformer = Pipeline([
            ('one_hot', cate_encoder)
            , ('pca', TruncatedSVD(n_components=50))
        ])
        xgb_sparrse_transformer = Pipeline([
            ('ordinal', OrdinalEncoder())
        ])
        self.transformer = ColumnTransformer(
            transformers=[
                ("lr_dense", lr_dense_transformer, criteo_dense_features),
                ("lr_sparse", lr_sparse_transformer, criteo_sparse_features),
            ]
            , remainder='drop'
        )
        self.xgb_transformer = ColumnTransformer(
            transformers=[
                ("xgb_sparse", xgb_sparrse_transformer, criteo_sparse_features),
            ]
            , remainder='passthrough'
        )
        self.transformer.fit(df_for_encode_train.copy())
        self.xgb_transformer.fit(df_for_encode_train.copy())
        logging.info(f"LR transformer info: ")
        logging.info(self.transformer)
        logging.info(f"XGB transformer info: ")
        logging.info(self.xgb_transformer)

        if train_valid:
            train_X, test_X, train_y, test_y = train_test_split(self.xgb_transformer.transform(X.copy()), y, test_size=0.2)
            self.xgb.fit(X=train_X, y=train_y, verbose=True, eval_metric='logloss'
                         , eval_set=[[train_X, train_y], [test_X, test_y]])
            self._plot_eval_result()
        else:
            self.xgb.fit(X=X, y=y, verbose=True, eval_metric='logloss'
                         , eval_set=[[X, y]])
        leave_info = self.xgb.apply(self.xgb_transformer.transform(X.copy()))#[:, 70:]
        logging.info(f"leave dim {X.shape}")
        self.one_hot = OneHotEncoder()
        logging.info(f"one hot...")
        xgb_features = self.one_hot.fit_transform(leave_info)

        logging.info(f"finished one hot")
        lr_feature = self.transformer.transform(X.copy())#self.pipeline.fit_transform(X)
        xgb_features = sparse.csc_matrix(xgb_features)
        all_features = sparse.hstack([xgb_features, lr_feature])
        self.lr = LogisticRegression(
        penalty='l1', solver='saga', verbose=1)
        logging.info(f"Logistic regression training...")
        self.lr.fit(all_features, y)
        logging.info(f"Logistic regression train finished")

    def predict_proba(self, X) -> pd.DataFrame:
        # if self.lr is not None:
        # X = self.ordianl.transform(X)
        lr_feature = self.transformer.transform(X.copy())
        xgb_dense_features = self.xgb_transformer.transform(X.copy())
        leave_info = self.xgb.apply(xgb_dense_features)#[:, 70:]
        xgb_features = self.one_hot.transform(leave_info)
        xgb_features = sparse.csc_matrix(xgb_features)
        all_features = sparse.hstack([xgb_features, lr_feature])
        logging.info(f"finished one hot")
        res = self.lr.predict_proba(X=all_features)[:, 1]
        return res