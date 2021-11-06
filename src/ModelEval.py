# -*- coding: utf-8 -*-
# @File    : ModelEval.py
# @Author  : Hua Guo
# @Time    : 2021/11/6 上午11:12
# @Disc    :
import pandas as pd


class ModelEval:
    def __init__(self, ground_truth: pd.DataFrame, predicted: pd.DataFrame, joined_key=['user_id']):
        self.ground_truth = ground_truth
        self.predicted = predicted
        self.combine = self.predicted.merge(self.ground_truth, how='left', on=joined_key)
        self.combine['match'] = self.combine['movie_id_x'] == self.combine['movie_id_y']

    def get_hr(self, at_n=10):
        combine = self.combine.assign(
            rank=self.combine.sort_values('predict_prob', ascending=False).groupby('user_id').cumcount()+1
        )
        hr = round(combine[combine['rank']<=at_n]['match'].sum()/len(self.ground_truth), 3)
        return hr


