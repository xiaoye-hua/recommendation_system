# -*- coding: utf-8 -*-
# @File    : pca_utils.py
# @Author  : Hua Guo
# @Time    : 2021/11/4 上午6:43
# @Disc    :
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def plot_pca_res(train: pd.DataFrame, file_name=None) -> None:
    sc_V = MinMaxScaler()
    sc_V.fit(train)
    train = sc_V.transform(train)
    plt.figure(figsize=(25,6))
    pca = PCA().fit(train)
    plt.plot(range(1, train.shape[1]+1),np.cumsum(pca.explained_variance_ratio_), "bo-")
    plt.xlabel("Component Count")
    plt.ylabel("Variance Ratio")
    plt.xticks(range(1, train.shape[1]+1))
    plt.grid()
    if file_name is not None:
        plt.savefig(file_name, pad_inches='tight')
    plt.show()