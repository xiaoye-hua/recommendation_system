# -*- coding: utf-8 -*-
# @File    : ItemPopPipeline.py
# @Author  : Hua Guo
# @Time    : 2021/11/6 下午1:53
# @Disc    :
import numpy as np
import logging
logging.getLogger(__name__)

from src.Pipeline import BasePipeline


class ItemPopPipeline(BasePipeline):
    def __init__(self, model_path: str, model_training=False, **kwargs):
        super(ItemPopPipeline, self).__init__(model_path=model_path, model_training=model_training)
        self.pipeline = None
        self.model_file_name = "model.pkl"
        self.num_df = None
        self.item_col = 'movie_id'
        self.user_col = 'user_id'

    def train(self, X, y, train_params):
        assert self.item_col in list(X.columns)
        X = X.assign(
            label=y
        )
        X = X[X.label==1]
        self.num_df = X[[self.item_col, self.user_col]].groupby(self.item_col).agg({self.user_col: np.size}).reset_index().rename(columns={self.user_col: 'prob'})
        logging.info(f"The most popular Item:")
        logging.info(self.num_df.head())

    def predict_proba(self, X):
        assert self.item_col in list(X.columns)
        assert self.user_col in list(X.columns)
        X = X.merge(self.num_df, how='left', on=self.item_col)
        return X['prob']

    def save_pipeline(self):
        pass

    def load_pipeline(self):
        pass

    def eval(self, X, y, **kwargs):
        return 0.5
        # predict_prob = self.predict_proba(X)
        # fpr, tpr, thresholds = roc_curve(y, predict_prob[:, 1])
        # auc_score = auc(fpr, tpr)
        # return auc_score