# -*- coding: utf-8 -*-
# @File    : model_train.py
# @Author  : Hua Guo
# @Time    : 2021/11/5 下午10:37
# @Disc    :
import pandas as pd
import os
import logging
logging.getLogger(__name__)

from src.FeatureCreator.NCFFeatureCreator import NCFFeatureCreator
from src.Pipeline.NCFPipeline import NCFPipeline
from src.Pipeline.ItemPopPipeline import ItemPopPipeline
from scripts.train_config import user_item_feature_path, cleaned_data_dir, debug, debug_user_item_feature_path, debug_cleaned_dir
from src.config import csv_sep
from src.ModelEval import ModelEval
# from scripts.train_config import config_dict

# =============== config ===============
mark = 'ncf_1106'

pipeline_class = NCFPipeline
# pipeline_class = ItemPopPipeline
fc_class = NCFFeatureCreator

# pipeline_class = config_dict[mark]['pipeline']
# fc_class = config_dict[mark]['feature_creator']


model_path = os.path.join('model_training', mark)
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
    , 'batch_size': 256
    , 'epoches': 5
}
pipeline.train(X=features, y=features['rating'], train_params=train_params)


logging.info(f"Loading testing data ...")
test_data = pd.read_csv(os.path.join(train_data_path, 'candidate.csv'), sep=csv_sep)
eval_data = pd.read_csv(os.path.join(train_data_path, 'test.csv'), sep=csv_sep)
logging.info(f"Eval...")
test_features = fc.get_features(uid_item_df=test_data)
test_features['predict_prob'] = pipeline.predict_proba(test_features)
model_eval = ModelEval(ground_truth=eval_data, predicted=test_features[['user_id', 'movie_id', 'predict_prob']])

for n in [1, 5, 10]:
    hr = model_eval.get_hr(at_n=n)
    print(hr)







