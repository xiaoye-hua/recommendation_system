# -*- coding: utf-8 -*-
# @File    : plot_utils.py
# @Author  : Hua Guo
# @Time    : 2021/10/29 下午9:56
# @Disc    :
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from xgboost.sklearn import XGBModel
from typing import List
from sklearn.metrics import roc_curve, classification_report, roc_auc_score, confusion_matrix
from src.utils.confusion_matrix_pretty_print import pretty_plot_confusion_matrix


def count_plot(df: pd.DataFrame, col: str, xytext=(0, 0), show_details=True) -> None:
    '''
    custom count plot
    Args:
        df:
        col:
        xytext:

    Returns:

    '''
    ax = sns.countplot(data=df, x=col)
    if show_details:
        for bar in ax.patches:
            ax.annotate('%{:.2f}\n{:.0f}'.format(100*bar.get_height()/len(df),bar.get_height()), (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                        size=11, xytext=xytext,
                        textcoords='offset points')
    plt.show()


def plot_feature_importances(model: XGBModel, feature_cols: List[str], show_feature_num=10, figsize=(20, 10), fig_dir=None):
    """
    plot feature importance of xgboost model
    Args:
        model:
        feature_cols:
        show_feature_num:
        figsize:

    Returns:

    """
    feature_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)[:show_feature_num]
    plt.figure(figsize=figsize)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.title("Feature Importance")
    if fig_dir is not None:
        plt.savefig(os.path.join(fig_dir, 'feature_importance.png'))
    else:
        plt.show()


def plot_auc_plot(y_test: pd.DataFrame, pred_prob: pd.DataFrame, fig_dir=None) -> None:
    auc = roc_auc_score(y_test, pred_prob)
    false_positive_rate, true_positive_rate, thresolds = roc_curve(y_test, pred_prob)
    plt.figure(figsize=(5, 5), dpi=100)
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("AUC & ROC Curve")
    plt.plot(false_positive_rate, true_positive_rate, 'g')
    plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
    plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if fig_dir is not None:
        plt.savefig(os.path.join(fig_dir, 'auc_plot.png'))
    else:
        plt.show()


def binary_classification_eval(test_y: pd.DataFrame, predict_prob: pd.DataFrame, fig_dir=None) -> None:
    # plot_confusion_matrix(model, test_X, test_y, values_format='')
    optimal_threshold = get_optimal_threshold(y=test_y, y_score=predict_prob)
    test_label_optimal = [0 if ele < optimal_threshold else 1 for ele in predict_prob]
    print("*"*20)
    print(f"Optimal threshold: {optimal_threshold}")
    print("*"*20)
    print(classification_report(test_y, test_label_optimal))
    print("*"*20)
    # confusion matrix
    pretty_plot_confusion_matrix(df_cm=pd.DataFrame(confusion_matrix(test_y, test_label_optimal)), fig_dir=fig_dir)
    # auc
    y_test = test_y
    # y_pred = predict_prob
    plot_auc_plot(y_test=y_test, pred_prob=predict_prob, fig_dir=fig_dir)


def get_optimal_threshold(y: pd.DataFrame, y_score: pd.datetime) -> float:
    fpr, tpr, threshold = roc_curve(y_true=y, y_score=y_score)
    objective_func = abs(fpr + tpr - 1)
    idx = np.argmin(objective_func)
    optimal_threshold = threshold[idx]
    return optimal_threshold


def labels(ax, df, xytext=(0, 0)):
    for bar in ax.patches:
        ax.annotate('%{:.2f}\n{:.0f}'.format(100*bar.get_height()/len(df),bar.get_height()), (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                    size=11, xytext=xytext,
                    textcoords='offset points')


def cate_features_plot(df, col, target, target_binary=True, figsize=(20,6)):
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)

    plt.subplot(121)
    if target_binary:
        tmp = round(pd.crosstab(df[col], df[target], normalize='index'), 2)
        tmp = tmp.reset_index()
        # tmp.rename(columns={0: 'NoFraud', 1: 'Fraud'}, inplace=True)

    ax[0] = sns.countplot(x=col, data=df, hue=target,
                          order=np.sort(df[col].dropna().unique()),
                          )
    ax[0].tick_params(axis='x', rotation=90)
    labels(ax[0], df[col].dropna(), (0, 0))
    if target_binary:
        ax_twin = ax[0].twinx()
        # sns.set(rc={"lines.linewidth": 0.7})
        ax_twin = sns.pointplot(x=col, y=1, data=tmp, color='black', legend=False,
                                order=np.sort(df[col].dropna().unique()),
                                linewidth=0.1)

    ax[0].grid()

    plt.subplot(122)
    ax[1] = sns.countplot(x=df[col].dropna(),
                          order=np.sort(df[col].dropna().unique()),
                          )
    ax[1].tick_params(axis='x', rotation=90)
    labels(ax[1], df[col].dropna())
    plt.show()