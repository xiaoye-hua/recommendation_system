# -*- coding: utf-8 -*-
# @File    : DeepFMPipeline.py
# @Author  : Hua Guo
# @Time    : 2021/11/6 下午7:55
# @Disc    :
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat

from src.Pipeline import BasePipeline
from src.DataPreprocess.NewOrdinalEncoder import NewOrdinalEncoder
from src.DataPreprocess.NewMinMaxScaler import NewMinMaxSaler
from src.config import criteo_sparse_features, criteo_dense_features


class DeepFMPipeline(BasePipeline):
    def __init__(self, model_path: str, model_training=False, **kwargs):
        super(DeepFMPipeline, self).__init__(model_path=model_path, model_training=model_training, **kwargs)

    def train(self, X, y, train_params):
        pre_process_pipeline_lst = []
        df_for_encode_train = train_params['df_for_encode_train']
        batch_size = train_params['batch_size']
        epoches = train_params['epoches']

        # Info of all of the data
        transformer = NewOrdinalEncoder(category_cols=criteo_sparse_features)
        transformer.fit_transform(df_for_encode_train)
        # mms = MinMaxScaler(feature_range=(0, 1))
        mms = NewMinMaxSaler(dense_cols=criteo_dense_features, feature_range=[0, 1])
        df_for_encode_train = mms.fit_transform(df_for_encode_train)
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=df_for_encode_train[feat].max() + 1, embedding_dim=4)
                                  for i, feat in enumerate(criteo_sparse_features)] + [DenseFeat(feat, 1, )
                                                                                for feat in criteo_dense_features]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        # data transform
        X = transformer.transform(X=X)
        X = mms.transform(X)
        pre_process_pipeline_lst.extend(
           [ ('ordinal_transformer', transformer)
            , ("mms", mms)
        ])
        self.preprocess_pipeline = Pipeline(pre_process_pipeline_lst)
        # feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        train_model_input = self._process_train_data(X)
        train_label = y.values
        self.model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
        self.model.summary()
        self.model.compile("adam", "binary_crossentropy",
                      metrics=['binary_crossentropy', tf.keras.metrics.AUC()], )

        self.model.fit(train_model_input, train_label,
                        batch_size=batch_size,
                       epochs=epoches,
                       # verbose=2,
                       validation_split=0.2,
                       )
        print()

    def _process_train_data(self, X):
        train_model_input = {}
        for col in criteo_sparse_features+criteo_dense_features:
            train_model_input[col] = X[col]
        # print(train_model_input)
        return train_model_input

    def predict_proba(self, X):
        X = self.preprocess_pipeline.transform(X)
        trian_input = self._process_train_data(X)
        prob = self.model.predict(trian_input)
        return prob

    def save_pipeline(self):
        pass

    def load_pipeline(self):
        pass