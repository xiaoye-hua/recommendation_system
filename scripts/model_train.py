# -*- coding: utf-8 -*-
# @File    : model_train.py
# @Author  : Hua Guo
# @Time    : 2021/11/5 下午10:37
# @Disc    :
import pandas as pd
import os

from src.FeatureCreator.NCFFeatureCreator import NCFFeatureCreator
from src.Pipeline.NCFPipeline import NCFPipeline
from scripts.train_config import user_item_feature_path, cleaned_data_dir, debug, debug_user_item_feature_path, debug_cleaned_dir
from src.config import csv_sep

# =============== config ===============
pipeline_class = NCFPipeline
fc_class = NCFFeatureCreator
model_path = 'model_training/ncf_1106'


if debug:
    user_item_path = debug_user_item_feature_path
    train_data_path = debug_cleaned_dir
else:
    user_item_path = user_item_feature_path
    train_data_path = cleaned_data_dir
# ======================================


fc = fc_class(profile_feature_dir=user_item_path)
pipeline = pipeline_class(model_path=model_path, model_training=True)
train_data = pd.read_csv(os.path.join(train_data_path, 'train.csv'), sep=csv_sep)
features = fc.get_features(uid_item_df=train_data)
train_params = {
    "df_for_encode_train": features
    , 'batch_size': 64
    , 'epoches': 2
}
pipeline.train(X=features, y=features['rating'], train_params=train_params)