# Fern

NLP text processing toolkit

## 变量命名规则

为了方便定义，对容易分歧变量命名做如下约定：

1. 对于数据变量，同类型变量书写规则：
   - `data_train`, `data_val`
   - `label_train`, `label_val`
   
2. 对于指标变量，同类型变量书写规则：
    - `val_loss`, `val_acc`, `val_binary_acc`
    - `train_loss`, `train_acc`

3. 对于其他变量，按照`首先它属于a, 其次它属于b`规则命名变量名：`a_b`
   
    - `path_dataset`
