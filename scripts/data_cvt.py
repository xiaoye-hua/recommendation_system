# -*- coding: utf-8 -*-
# @File    : data_cvt.py
# @Author  : Hua Guo
# @Time    : 2021/10/28 上午11:44
# @Disc    :
import pandas as pd
import os

from scripts.train_config import raw_data_dir, cleaned_data_dir
from src.DataConvert.NCFDataConvertor import NCFDataConvertor


model_params = {

}
convertor = NCFDataConvertor(input_dir=raw_data_dir, output_dir=cleaned_data_dir, test_ratio=0.2, split_mode='random')
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