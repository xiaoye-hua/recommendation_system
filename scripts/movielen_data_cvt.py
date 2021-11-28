# -*- coding: utf-8 -*-
# @File    : movielen_data_cvt.py
# @Author  : Hua Guo
# @Time    : 2021/10/28 上午11:44
# @Disc    :
import logging

from scripts.train_config import user_item_feature_path, cleaned_data_dir
from src.DataConvert.NCFDataConvertor import NCFDataConvertor
from scripts.train_config import debug_user_item_feature_path, debug_cleaned_dir
from scripts.movielens_config import debug, mark, default_data_cvt_config, all_config

logging.basicConfig(level='INFO',
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    )


data_cvt_config: dict = all_config[mark].get('data_cvt_config', default_data_cvt_config)

if debug:
    user_item_path = debug_user_item_feature_path
    train_data_path = debug_cleaned_dir
else:
    user_item_path = user_item_feature_path
    train_data_path = data_cvt_config['train_data_path']

logging.info(f'data convert config:')
logging.info(data_cvt_config)
convertor = NCFDataConvertor(input_dir=user_item_path,
                             output_dir=train_data_path
                             , split_method=data_cvt_config['split_method']
                             , test_ratio=data_cvt_config['test_ratio']
                             , split_mode=data_cvt_config['split_mode']
                             , negsample=data_cvt_config['negsample']
                             )
convertor.convert()



# unames = ['user_id','gender','age','occupation','zip']
# user = pd.read_csv(os.path.join(raw_data_dir, 'users.dat'),sep='::',header=None,names=unames)
# rnames = ['user_id','movie_id','rating','timestamp']
# ratings = pd.read_csv(os.path.join(raw_data_dir, 'ratings.dat'),sep='::',header=None,names=rnames)
# mnames = ['movie_id','title','genres']
# movies = pd.read_csv(os.path.join(raw_data_dir, 'movies.dat'),sep='::',header=None,names=mnames)
# data = pd.merge(pd.merge(ratings,movies),user)#.iloc[:10000]
# print(f"data shape: {data.shape}")
# data.to_csv(os.path.join(cleaned_data_dir, cleaned_file_name), index=False)