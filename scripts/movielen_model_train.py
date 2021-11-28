# -*- coding: utf-8 -*-
# @File    : movielen_model_train.py
# @Author  : Hua Guo
# @Time    : 2021/11/5 下午10:37
# @Disc    :
import pandas as pd
import os
import logging
logging.getLogger(__name__)

from src.Pipeline.ItemPopPipeline import ItemPopPipeline
from scripts.train_config import user_item_feature_path, cleaned_data_dir, debug_user_item_feature_path, debug_cleaned_dir
from src.config import csv_sep
from src.ModelEval import ModelEval
from scripts.movielens_config import all_config, debug, mark

# =============== config ===============
logging.basicConfig(level='INFO',
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    )

pipeline_class = all_config[mark]['pipeline_class']
# pipeline_class = ItemPopPipeline
fc_class = all_config[mark]['feature_creator_class']
if debug:
    user_item_path = debug_user_item_feature_path
    train_data_path = debug_cleaned_dir
    model_path = os.path.join('model_training/debug', mark)
else:
    user_item_path = user_item_feature_path
    train_data_path = cleaned_data_dir
    model_path = os.path.join('model_training', mark)
# ======================================
fc = fc_class(profile_feature_dir=user_item_path)
pipeline = pipeline_class(model_path=model_path, model_training=True)
train_data = pd.read_csv(os.path.join(train_data_path, 'train.csv'), sep=csv_sep)
features = fc.get_features(df=train_data)
train_params = {
    "df_for_encode_train": features.copy()
    , 'batch_size': 256
    , 'epoches': 1
}
pipeline.train(X=features.copy(), y=features['rating'], train_params=train_params)
logging.info(f"Evalation on training data...")
pipeline.eval(X=features.copy(), y=features['rating'])
pipeline.save_pipeline()
logging.info(f"Loading testing data ...")
test_data = pd.read_csv(os.path.join(train_data_path, 'candidate.csv'), sep=csv_sep)
eval_data = pd.read_csv(os.path.join(train_data_path, 'test.csv'), sep=csv_sep)
logging.info(f"Eval...")
test_features = fc.get_features(df=test_data)
test_features['predict_prob'] = pipeline.predict_proba(test_features)
model_eval = ModelEval(ground_truth=eval_data, predicted=test_features[['user_id', 'movie_id', 'predict_prob']])

for n in [1, 5, 10]:
    hr = model_eval.get_hr(at_n=n)
    print(hr)







