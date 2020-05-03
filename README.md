# FERN

>  **English Version** | [中文版](./README_ZH.md)

Fern defines a model development structure control for NLP. With the help of Fern, the text preprocessing, model building and model training can be implemented quickly. These modules contain the following functions:

1. Text preprocessing: data downloader, data cleaner, data transformer and data splitter
2. Model building: model saving, loading and architecture printing
3. Model training: step /epochs training and evaluation, evaluation function setting, loss function setting and label weight setting

The design purpose of Fern is mainly to solve the problem of too much repetitive code in different NLP projects and reduce the flow code, so as to avoid random bugs in the process data interaction

## INSTALL

1. Install from  `pypi`

   ```shell
   $ pip install Fern2
   ```

2. Install from source code

   ```shell
   $ pip install -e git+https://github.com/Jasonsey/Fern.git@develop
   ```

## TUTORIAL

This is a quick tutorial that covers the basics of all classes. For more usage methods, it is recommended to see the instructions for the functions in the source code

### DATA PREPARATION

1. Data download

   ```python
   from fern.utils.data import BaseDownloader
   
   
   loader = BaseDownloader(host=config.HOST, user=config.USER, password=config.PASSWORD)
   loader.read_msssql(sql=config.SQL)
   loader.save(config.SOURCE_PATH)
   ```

2. Load the downloaded data from disk

   ```python
   loader.load(config.SOURCE_PATH)
   ```

3. Data cleaning

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

4. Data transforming

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

5. Data segmentation

   ```python
   from fern.utils.data import BaseSplitter
   
   
   splitter = BaseSplitter(rate_val=config.RATE_VAL)
   splitter.split(transformer.data)
   splitter.save(config.SPLIT_DATA)
   ```

### MODEL SEARCH

1. Configure the list of models to be searched


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

2. Searching the best model

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

### TRAINING THE BEST MODEL

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

## VARIABLE NAMING RULE

In order to facilitate the definition, the following convention is made for the naming of easily divergent variables：

1. For data variables, write rules for variables of the same type：
   - `data_train`, `data_val`
   - `label_train`, `label_val`

2. For indicator variables, write rules for variables of the same type：
   - `val_loss`, `val_acc`, `val_binary_acc`
   - `train_loss`, `train_acc`

3. For other variables, according to the rule that first it belongs to a and second it belongs to b：`a_b`

   - `path_dataset`

## CNAGE LOG

[CHANGE LOG](./CHANGELOG.md)

