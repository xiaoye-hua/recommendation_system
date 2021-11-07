# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Author  : Hua Guo
# @Time    : 2021/10/29 上午9:21
# @Disc    :

from abc import ABCMeta, abstractmethod
import os
import pandas as pd

from src.utils.plot_utils import binary_classification_eval


class BasePipeline(metaclass=ABCMeta):
    def __init__(self, model_path: str, model_training=False, **kwargs):
        self.model_training = model_training
        self.model_path = model_path
        self._check_dir(self.model_path)
        self.eval_result_path = os.path.join(self.model_path, 'eval')

    @abstractmethod
    def train(self, X, y, train_params):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    @abstractmethod
    def save_pipeline(self):
        pass

    @abstractmethod
    def load_pipeline(self):
        pass

    def _check_dir(self, directory):
        if not os.path.isdir(directory):
            os.makedirs(directory)

    def eval(self, X: pd.DataFrame, y: pd.DataFrame, default_fig_dir=None) -> None:
        if default_fig_dir is None:
            fig_dir = self.eval_result_path
        else:
            fig_dir = default_fig_dir
        self._check_dir(fig_dir)
        predict_prob = self.predict_proba(X)
        binary_classification_eval(test_y=y, predict_prob=predict_prob, fig_dir=fig_dir)