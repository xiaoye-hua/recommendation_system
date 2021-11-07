# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Author  : Hua Guo
# @Time    : 2021/10/29 上午9:00
# @Disc    :
from abc import ABCMeta, abstractmethod


class BaseFeatureCreator(metaclass=ABCMeta):

    @abstractmethod
    def get_features(self, df, **kwargs):
        pass