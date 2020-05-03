# FERN

>  **中文版** | [English Version](./README.md)

Fern用于NLP的模型开发结构控制。通过它可以控制文本预处理、模型搭建、训练器，这几个模块都包含如下功能：

1. 文本预处理：数据下载、数据清洗、数据转换和数据分割
2. 模型搭建：模型保存与加载、模型架构打印
3. 模型训练：单步/epoch训练与评估、评估函数设置、损失函数设置、label权重设置

Fern的设计目的主要为了解决不同NLP工程中重复代码过多问题，减少流程性代码，从而避免数据交互过程中的随机bug出现

## 安装

1. 从 `pypi` 安装

   ```shell
   $ pip install Fern2
   ```

2. 从源码安装

   ```shell
   $ pip install -e git+https://github.com/Jasonsey/Fern.git@develop
   ```

## 使用教程

这是一个快速上手的教程，包含了所有类的基础使用。更多使用方法，建议查看源码中函数的使用说明

### 数据准备

1. 数据下载

   ```python
   from fern.utils.data import BaseDownloader
   
   
   loader = BaseDownloader(host=config.HOST, user=config.USER, password=config.PASSWORD)
   loader.read_msssql(sql=config.SQL)
   loader.save(config.SOURCE_PATH)
   ```

2. 从硬盘加载下载好的数据

   ```python
   loader.load(config.SOURCE_PATH)
   ```

3. 数据清洗

   ```python
   from fern.utils.data import BaseCleaner
   
   
   class DataCleaner(BaseCleaner):
       def clean_label(self, row):
           return row['LABEL']
   
       def clean_data(self, row):
       		data = row['DATA']
           res = do_clean(data)
           return res
   
     
   cleaner = DataCleaner(stop_words=config.STOP_WORDS, user_words=config.USER_WORDS)
   cleaner.clean(loader.data)
   ```

4. 数据转换

   ```python
   from fern.utils.data import BaseTransformer
   
   
   class DataTransformer(BaseTransformer):
       def transform_label(self, label):
           res = np.zeros([1] + self.output_shape, np.float32)
           for i in range(len(str(label))):
               number = int(str(label)[i])
               res[:, i, number] = 1.0
           return res
   
   
   transformer = DataTransformer(
       data=cleaner.data,
       word_path=config.WORDS_LIBRARY,
       min_len=config.MIN_SEQ_LEN,
       max_len=config.MAX_SEQ_LEN,
       min_freq=config.MAX_WORD_FREQ,
       output_shape=config.OUTPUT_SHAPE,
       filter_data=True)
   transformer.transform(data=cleaner.data)
   transformer.save(config.TRANSFORMED_DATA)
   ```

5. 数据分割

   ```python
   from fern.utils.data import BaseSplitter
   
   
   splitter = BaseSplitter(rate_val=config.RATE_VAL)
   splitter.split(transformer.data)
   splitter.save(config.SPLIT_DATA)
   ```

### 模型搜索

1. 配置待搜索模型列表


  ```python
  from fern.utils.model import BaseModel
  
  class TextConv1D_1(BaseModel):
      def build(self):
          inp = layers.Input(shape=(self.max_seq_len,))
          x = layers.Dense(12)(x)
          oup = layers.Activation('softmax')(x)
          model = Model(
              inputs=inp,
              outputs=oup,
              name=self.name)
          return model
  
  class TextConv1D_2(BaseModel):
      def build(self):
          inp = layers.Input(shape=(self.max_seq_len,))
          x = layers.Dense(24)(x)
          oup = layers.Activation('softmax')(x)
          model = Model(
            	inputs=inp,
            	outputs=oup,
            	name=self.name)
          return model
  
  UNOPTIMIZED_MODELS = [TextConv1D_1, TextConv1D_2]
  ```

2. 搜索模型

   ```python
   from fern.utils.train import BaseTrainer
   
   
   best_score = 0
   best_epoch = 0
   best_model = ''
   
   for model in UNOPTIMIZED_MODELS:
   		tf.keras.backend.clear_session()
   		try:
       	my_model = model(
             output_shape=config.OUTPUT_SHAPE, 
             max_seq_len=config.MAX_SEQ_LEN, 
             library_len=library_len)
       	trainer = BaseTrainer(
             model=my_model,
             path_data=config.SPLIT_DATA,
             lr=config.LR,
             batch_size=config.BATCH_SIZE)
       score, epoch = trainer.train(
           config.EPOCHS,
           early_stop=config.EARLY_STOP)
       if score > best_score:
         best_score = score
         best_epoch = epoch
         best_model = my_model.name
   
   print(f'Best Model: {best_model}, Best Score: {best_score}, Best Epoch: {best_epoch}')
   ```

### 训练最佳模型

```python
my_model = UNOPTIMIZED_MODELS[0](
    output_shape=config.OUTPUT_SHAPE, 
    max_seq_len=config.MAX_SEQ_LEN,
    library_len=library_len)

trainer = ModelTrainer(
    model=my_model,
    path_data=config.SPLIT_DATA,
    lr=config.LR,
    batch_size=config.BATCH_SIZE)

_ = trainer.train(config.BEST_EPOCH, mode='server')
trainer.save(config.MODEL_PATH)
```

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

## 版本变更日志

[CHANGE LOG](./CHANGELOG.md)