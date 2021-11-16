# -*- coding: utf-8 -*-
# @File    : model_train.py
# @Author  : Hua Guo
# @Time    : 2021/11/5 下午10:37
# @Disc    :
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
logging.getLogger(__name__)

# from src.FeatureCreator.NCFFeatureCreator import NCFFeatureCreator
# from src.Pipeline.NCFPipeline import NCFPipeline
# from src.Pipeline.ItemPopPipeline import ItemPopPipeline
from scripts.train_config import user_item_feature_path, cleaned_data_dir, debug, debug_user_item_feature_path, debug_cleaned_dir
# from src.config import csv_sep
# from src.ModelEval import ModelEval
from scripts.train_config import config_dict
from src.config import criteo_csv_sep, criteo_df_cols, criteo_target_col

# =============== config ===============
# mark = 'deepFM_1107_criteo'
# mark = 'LR_1108_criteo'
# mark = 'XGB_1108_criteo'
# mark = 'xgblr_1109_criteo'
# mark = 'LR_one_hot_1108_criteo'
mark = 'xgblr_1116_criteo'
logging.info(f"Mark: {mark}")
logging.info(config_dict[mark])



# pipeline_class = NCFPipeline
# pipeline_class = ItemPopPipeline
# fc_class = NCFFeatureCreator

pipeline_class = config_dict[mark]['pipeline']
fc_class = config_dict[mark]['feature_creator']
one_hot = config_dict[mark].get('one_hot', False)
# dense_bin = config_dict[mark].get("dense_bin", False)
# dense_standard = config_dict[mark].get('dense_standard', False)


model_path = os.path.join('model_training', mark)
if debug:
    # user_item_path = debug_user_item_feature_path
    # train_data_path = debug_cleaned_dir
    data_dir = config_dict[mark]['debug_data_dir']
    model_path = 'model_training/debug'
else:
    data_dir = config_dict[mark]['production_data_dir']


    # user_item_path = user_item_feature_path
    # train_data_path = cleaned_data_dir
# ======================================


fc = fc_class()
model_params = {
    'scale_pos_weight': 3
}
pipeline = pipeline_class(model_path=model_path, model_training=True, model_params=model_params)
# train_data = pd.read_csv(data_dir, sep=
logging.info('Loading raw data...')
train_data = pd.read_csv(data_dir,
                 sep=criteo_csv_sep,
                 header=None
                , names=criteo_df_cols
                , nrows=2000000
                         )
logging.info(f"Feature creating...")
features, feature_cols = fc.get_features(df=train_data)

train, test = train_test_split(features, test_size=0.2)
logging.info(f"Train data + label shape: {train.shape}")
logging.info(f"Test data + label shape: {test.shape}")

train_params = {
    "df_for_encode_train": features.copy()[feature_cols]
    , 'batch_size': 256
    , 'epoches': 2
    , 'train_valid': True
    , 'one_hot': one_hot
    , 'dense_bin': config_dict[mark].get("dense_bin", False)
    , 'dense_standard': config_dict[mark].get('dense_standard', False)
    # , 'lr': True
}
pipeline.train(X=train.copy()[feature_cols], y=train[criteo_target_col], train_params=train_params)
pipeline.save_pipeline()
logging.info(f"Evaling...")
logging.info(f"Eval...")
pipeline.eval(
    X=test[feature_cols], y=test[criteo_target_col]
    # X=train.copy()
    # , y=train[criteo_target_col]
              )
logging.info(f"Eval finished")
# pipeline.eval(
#     X=features[feature_cols], y=features[criteo_target_col]
#     # X=train.copy()
#     # , y=train[criteo_target_col]
#               )


# logging.info(f"Loading testing data ...")
# test_data = pd.read_csv(os.path.join(train_data_path, 'candidate.csv'), sep=csv_sep)
# eval_data = pd.read_csv(os.path.join(train_data_path, 'test.csv'), sep=csv_sep)
# logging.info(f"Eval...")
# test['predict_prob'] = pipeline.predict_proba(test_features)
# model_eval = ModelEval(ground_truth=eval_data, predicted=test_features[['user_id', 'movie_id', 'predict_prob']])
#
# for n in [1, 5, 10]:
#     hr = model_eval.get_hr(at_n=n)
#     print(hr)







