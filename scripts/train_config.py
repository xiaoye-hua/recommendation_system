# -*- coding: utf-8 -*-
# @File    : train_config.py
# @Author  : Hua Guo
# @Time    : 2021/10/28 上午11:45
# @Disc    :
import logging
from src.FeatureCreator.NCFFeatureCreator import NCFFeatureCreator
from src.FeatureCreator.DeepFMFeatureCreator import DeepFMFeatureCreator
from src.Pipeline.DeepFMPipeline import DeepFMPipeline
from src.Pipeline.NCFPipeline import NCFPipeline
from src.Pipeline.ItemPopPipeline import ItemPopPipeline

# ============ General Config =====================
debug = False
logging.basicConfig(level='INFO',
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',)


# =============== Train config ===================
config_dict = {
    'deepFM_1107_criteo': {
        'feature_creator': DeepFMFeatureCreator
        , 'pipeline': DeepFMPipeline
        , 'debug_data_dir': 'data/debug/debug_criteo_data/train.txt'
        , 'production_data_dir': 'data/raw_criteo_data/train.txt'
    }
}

# ============ Config for data convert ===========
debug_user_item_feature_path = 'data/debug/raw'
debug_cleaned_dir = 'data/debug/cleaned'
user_item_feature_path = 'data/ml-1m'
cleaned_data_dir = 'data/ml-1m_cleaned'
# cleaned_file_name = 'movielens.txt'


# ==============================================