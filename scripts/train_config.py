# -*- coding: utf-8 -*-
# @File    : train_config.py
# @Author  : Hua Guo
# @Time    : 2021/10/28 上午11:45
# @Disc    :
import logging

# ============ General Config =====================
debug = False
logging.basicConfig(level='INFO',
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',)

# ============ Config for data convert ===========
debug_user_item_feature_path = 'data/debug/raw'
debug_cleaned_dir = 'data/debug/cleaned'
user_item_feature_path = 'data/ml-1m'
cleaned_data_dir = 'data/ml-1m_cleaned'
# cleaned_file_name = 'movielens.txt'


# ==============================================