# -*- coding: utf-8 -*-
# @File    : movielens_config.py
# @Author  : Hua Guo
# @Disc    :
from src.Pipeline.NCFPipeline import NCFPipeline
from src.FeatureCreator.NCFFeatureCreator import NCFFeatureCreator

# ======= Config =========
debug = True
mark = 'v0_ncf_1127'


# ======= Config details ======
train_config = {
    "v0_ncf_1127": {
        'pipeline_class': NCFPipeline
        , 'feature_creator_class': NCFFeatureCreator
        ,
    }
}