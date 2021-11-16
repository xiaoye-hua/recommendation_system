# -*- coding: utf-8 -*-
# @File    : XGBoostLRPipeline.py
# @Author  : Hua Guo
# @Disc    :
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
import logging
logging.getLogger(__name__)
from sklearn.linear_model import LogisticRegression

from src.Pipeline import BasePipeline
from src.utils.plot_utils import plot_feature_importances, binary_classification_eval
from src.DataPreprocess.NewOrdinalEncoder import NewOrdinalEncoder
from src.Pipeline.XGBoostPipeline import XGBoostPipeline
from src.DataPreprocess.XGBoostLRDataProcess import XGBoostLRDataProcess


class XGBoostLRPipeline(XGBoostPipeline):
    def train(self, X: pd.DataFrame, y: pd.DataFrame, train_params: dict) -> None:
        pipeline_lst = []

        df_for_encode_train = train_params['df_for_encode_train']
        train_valid = train_params.get("train_valid", False)
        transformer = NewOrdinalEncoder(category_cols=self.cate_encode_cols)
        xgb_tranformer  = XGBoostLRDataProcess( #xgb_model=self.xgb, #train_params=train_params
             )
        lr = LogisticRegression(penalty='l1', solver='saga', verbose=1)
        transformer.fit(df_for_encode_train.copy())
        X = transformer.transform(X=X)
        logging.info(f"xgb_transformer fitting ...")
        X = xgb_tranformer.fit_transform(X=X, y=y)
        logging.info("xgb_transformer finished")
        lr.fit(X, y)
        pipeline_lst.append(("new_ordinal_transformer", transformer))
        pipeline_lst.append(
            ('xgblr_dataprocess', xgb_tranformer)
        )
        pipeline_lst.append(
            ('lr', lr)
        )
        self.pipeline = Pipeline(pipeline_lst)
        logging.info(f"Pipeline info: ")
        logging.info(self.pipeline)

    def predict_proba(self, X) -> pd.DataFrame:
        res = self.pipeline.predict_proba(X=X)[:, 1]
        return res
