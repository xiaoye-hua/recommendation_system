# -*- coding: utf-8 -*-
# @File    : movielens_config.py
# @Author  : Hua Guo
# @Disc    :
from src.Pipeline.NCFPipeline import NCFPipeline
from src.Pipeline.ItemPopPipeline import ItemPopPipeline
from src.FeatureCreator.NCFFeatureCreator import NCFFeatureCreator

# ======= Config =========
debug = True
mark = 'v0_itemPop_1128'

# default config for data convert
default_data_cvt_config ={
        'train_data_path': 'data/ml-1m_cleaned'
        , 'split_method': 'leave_one_out'
        , 'test_ratio': 0.2
        , 'split_mode': 'random'
        , 'negsample': 410
    }

# ======= Config details ======
all_config = {
    "v0_ncf_1127": {
        'pipeline_class': NCFPipeline
        , 'feature_creator_class': NCFFeatureCreator
        ,
    }
    , "v0_itemPop_1128": {
        'pipeline_class': ItemPopPipeline
        , 'feature_creator_class': NCFFeatureCreator
        , 'data_cvt_config': {
            'train_data_path': 'data/raw_data/movielen_seq_aware'
            , 'split_method': 'frac_split'
            , 'test_ratio': 0.2
            , 'split_mode': 'seq_aware'
            , 'negsample': 3
        }
    }
}

