# -*- coding: utf-8 -*-
# @File    : config.py
# @Author  : Hua Guo
# @Time    : 2021/10/28 下午10:37
# @Disc    :

csv_sep = '\t'
origin_data_sep = "::"


# debug_criteo_data dataset
criteo_csv_sep = '\t'
criteo_target_col = 'label'
criteo_sparse_features = ['C' + str(i) for i in range(1, 27)]
criteo_dense_features = ['I' + str(i) for i in range(1, 14)]
criteo_df_cols = [criteo_target_col] + criteo_dense_features + criteo_sparse_features

# movie len datasets
origin_user_cols= ['user_id','gender','age','occupation','zip']
origin_rating_cols = ['user_id','movie_id','rating','timestamp']
orgin_movie_cols = ['movie_id','title','genres']