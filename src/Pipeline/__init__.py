# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Author  : Hua Guo
# @Time    : 2021/10/29 上午9:21
# @Disc    :

from abc import ABCMeta, abstractmethod

import os


class BasePipeline(metaclass=ABCMeta):
    def __init__(self, model_path: str, model_training=False, **kwargs):
        self.model_training = model_training
        self.model_path = model_path
        self._check_dir(self.model_path)

    @abstractmethod
    def train(self, X, y, train_params):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    @abstractmethod
    def eval(self, X, y):
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