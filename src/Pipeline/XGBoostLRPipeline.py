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
        transformer.fit(df_for_encode_train)
        X = transformer.transform(X=X)
        pipeline_lst.append(("new_ordinal_transformer", transformer))
        self.ordianl = transformer
        if train_params.get('pca_component_num', False):
            pca_component_num = train_params['pca_component_num']
            self.pca_component_num = pca_component_num
            min_max = MinMaxScaler()
            pca = PCA(n_components=pca_component_num)
            X = min_max.fit_transform(X)
            X = pca.fit_transform(X)
            pipeline_lst.extend(
                [
                    ('min_max', min_max)
                    , ('pca', pca)
                ]
            )
        self.xgb = XGBClassifier(**self.model_params)
        print(f"Model params are {self.xgb.get_params()}")

        # if train_valid:
        #     train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
        #     train_params.update(
        #         {
        #             'verbose': True
        #             , 'eval_metric': 'logloss'
        #             , 'eval_set': [[train_X, train_y], [test_X, test_y]]
        #         }
        #     )
        # pipeline_lst.append(
        #     ('xgblr_dataprocess', XGBoostLRDataProcess(xgb_model=self.xgb, train_params=train_params))
        # )
        # pipeline_lst.append(
        #     ('lr', LogisticRegression(penalty='l1', solver='saga', verbose=1))
        # )
        # self.pipeline = Pipeline(pipeline_lst)
        # self.pipeline.fit(X, y)



        if train_valid:
            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
            # for trans in pipeline_lst:
            #     eval_X = trans[1].transform(eval_X)
            self.xgb.fit(X=X, y=y, verbose=True, eval_metric='logloss'
                         , eval_set=[[train_X, train_y], [test_X, test_y]])
            self._plot_eval_result()
            train_params.update(
                {
                    'verbose': True
                    , 'eval_metric': 'logloss'
                    , 'eval_set': [[train_X, train_y], [test_X, test_y]]
                }
            )
        else:
            self.xgb.fit(X=X, y=y, verbose=True, eval_metric='logloss'
                         , eval_set=[[X, y]])
        pipeline_lst.append(('model', self.xgb))
        self.pipeline = Pipeline(pipeline_lst)
        X = self.xgb.apply(X)#[:, 70:]
        logging.info(f"leave dim {X.shape}")
        self.one_hot = OneHotEncoder()
        logging.info(f"one hot...")
        X = self.one_hot.fit_transform(X)
        logging.info(f"finished one hot")
        self.lr = LogisticRegression(
        penalty='l1', solver='saga', verbose=1)
        logging.info(f"Logistic regression training...")
        self.lr.fit(X, y)
        logging.info(f"Logistic regression train finished")

    def predict_proba(self, X) -> pd.DataFrame:
        if self.lr is not None:
            X = self.ordianl.transform(X)
            X = self.xgb.apply(X)#[:, 70:]
            logging.info(f"leave dim {X.shape}")
            logging.info(f"one hot...")
            X = self.one_hot.transform(X)
            logging.info(f"finished one hot")
            res = self.lr.predict_proba(X=X)[:, 1]
        else:
            res = self.pipeline.predict_proba(X=X)[:, 1]
        return res