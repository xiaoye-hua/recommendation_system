# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Author  : Hua Guo
# @Time    : 2021/10/28 下午4:05
# @Disc    :
import os
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

from src.config import csv_sep, origin_rating_cols


class BaseDataConvertor(metaclass=ABCMeta):

    @abstractmethod
    def convert(self):
        pass

    def _check_dir(self, directory):
        if not os.path.isdir(directory):
            os.makedirs(directory)


def read_data_ml100k(raw_data_dir):
    data = pd.read_csv(os.path.join(raw_data_dir, 'ratings.dat'),sep='::',header=None,names=origin_rating_cols)
    num_users = data.user_id.unique().shape[0]
    num_items = data['movie_id'].unique().shape[0]
    return data, num_users, num_items

#@save
def split_data_ml100k(data, num_users, split_mode='random',
                      test_ratio=0.1):
    """Split the dataset in random mode or seq-aware mode."""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [
            True if x == 1 else False
            for x in np.random.uniform(0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data


def get_train_test_data(raw_data_dir: str, out_put_dir: str, split_mode='random',
                      test_ratio=0.1) -> None:
    data, num_users, num_items = read_data_ml100k(raw_data_dir=raw_data_dir)
    train_data, test_data = split_data_ml100k(data, num_users,
                                              split_mode, test_ratio)
    train_data.to_csv(os.path.join(out_put_dir, 'explicit_train.csv'), index=False, sep=csv_sep)
    test_data.to_csv(os.path.join(out_put_dir, 'test.csv'), index=False, sep=csv_sep)