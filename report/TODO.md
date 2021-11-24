# TODO


## Bug

It might caused by `deepctr` package, after loading the `savedModel` file, the name of input layer changed to 'input_1', 'input_2'...'input_n'. I've tried [low-level tf.saved_model API](https://www.tensorflow.org/guide/saved_model), the loaded model have the original layer name, but it does not have the `predict` api. As a result, I changed to a hybird way (refer to [](../src/Pipeline/DeepFMPipeline.py)):

```python
def _process_train_data(self, X):
    train_model_input = {}
    if self.model_training:
        for col in criteo_sparse_features+criteo_dense_features:
            train_model_input[col] = X[col]
    else:
        for idx, col in enumerate(criteo_sparse_features+criteo_dense_features):
            target_col = f'input_{idx+1}'
            train_model_input[target_col] = X[col]
    return train_model_input
```