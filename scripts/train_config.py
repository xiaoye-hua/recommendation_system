# -*- coding: utf-8 -*-
# @File    : train_config.py
# @Author  : Hua Guo
# @Time    : 2021/10/28 上午11:45
# @Disc    :
import logging
from src.FeatureCreator.NCFFeatureCreator import NCFFeatureCreator
from src.FeatureCreator.DeepFMFeatureCreator import DeepFMFeatureCreator
from src.Pipeline.DeepFMPipeline import DeepFMPipeline
from src.Pipeline.WideDeepPipeline import WideDeepPipeline
from src.Pipeline.LogisticRegressionPipeline import LogisticRegressionPipeline
from src.Pipeline.XGBoostPipeline import XGBoostPipeline
from src.Pipeline.XGBoostLRPipeline import XGBoostLRPipeline
from src.Pipeline.NCFPipeline import NCFPipeline
from src.Pipeline.ItemPopPipeline import ItemPopPipeline

# ============ General Config =====================
debug = True
logging.basicConfig(level='INFO',
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    # filename=os.path.join(model_path, 'train.log')
                    )



# =============== Train config ===================
config_dict = {
    'deepFM_1107_criteo': {
        'feature_creator': DeepFMFeatureCreator
        , 'pipeline': DeepFMPipeline
        , 'debug_data_dir': 'data/debug/debug_criteo_data/train.txt'
        , 'production_data_dir': 'data/raw_criteo_data/train.txt'
    }
    ,    'WDL_1107_criteo': {
        'feature_creator': DeepFMFeatureCreator
        , 'pipeline': WideDeepPipeline
        , 'debug_data_dir': 'data/debug/debug_criteo_data/train.txt'
        , 'production_data_dir': 'data/raw_criteo_data/train.txt'
    }
    ,  'LR_1108_criteo': {
        'feature_creator': DeepFMFeatureCreator
        , 'pipeline': LogisticRegressionPipeline
        , 'debug_data_dir': 'data/debug/debug_criteo_data/train.txt'
        , 'production_data_dir': 'data/raw_criteo_data/train.txt'
    }
    , 'LR_one_hot_1108_criteo': {
        'feature_creator': DeepFMFeatureCreator
        , 'pipeline': LogisticRegressionPipeline
        , 'one_hot': True
        , 'debug_data_dir': 'data/debug/debug_criteo_data/train.txt'
        , 'production_data_dir': 'data/raw_criteo_data/train.txt'
    }
    ,    'XGB_1108_criteo': {
        'feature_creator': DeepFMFeatureCreator
        , 'pipeline': XGBoostPipeline
        , 'debug_data_dir': 'data/debug/debug_criteo_data/train.txt'
        , 'production_data_dir': 'data/raw_criteo_data/train.txt'
    }
    , 'xgblr_1109_criteo': {
        'feature_creator': DeepFMFeatureCreator
        , 'pipeline': XGBoostPipeline
        , 'debug_data_dir': 'data/debug/debug_criteo_data/train.txt'
        , 'production_data_dir': 'data/raw_criteo_data/train.txt'
    }
    , 'LR_1116_criteo_DenseBinStand_CateOnehotSVD': {
        'feature_creator': DeepFMFeatureCreator
        , 'pipeline': LogisticRegressionPipeline
        , 'one_hot': True
        , 'dense_bin': True
        , 'dense_standard': True
        , 'debug_data_dir': 'data/debug/debug_criteo_data/train.txt'
        , 'production_data_dir': 'data/raw_criteo_data/train.txt'
    }
    ,  'LR_1116_criteo_DenseBin_CateOnehotSVD': {
        'feature_creator': DeepFMFeatureCreator
        , 'pipeline': LogisticRegressionPipeline
        , 'one_hot': True
        , 'dense_bin': True
        , 'dense_standard': False
        , 'debug_data_dir': 'data/debug/debug_criteo_data/train.txt'
        , 'production_data_dir': 'data/raw_criteo_data/train.txt'
    }
    # this set of params does not work right now.
    , 'xgblr_1116_criteo': {
        'feature_creator': DeepFMFeatureCreator
        , 'pipeline': XGBoostLRPipeline
        , 'debug_data_dir': 'data/debug/debug_criteo_data/train.txt'
        , 'production_data_dir': 'data/raw_criteo_data/train.txt'
    }
    , 'xgblr3_1116_criteo': {
        'feature_creator': DeepFMFeatureCreator
        , 'pipeline': XGBoostLRPipeline
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