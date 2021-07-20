# CHANGE LOG

## Version 1.0

- 1.1.1 (2021-07-18)
    - Fix Bug:
        - 修复Optional的错误
        - 默认的logger不允许从父类中继承handler
        - 修复分割数据集时, 大于1浮点数问题

- 1.1.0 (2021-07-14)
    - New Feature:
        - 添加Bert token方法
        - 添加yaml文件读取工具
        - 添加keras模型快速编译函数
        - 添加keras模型基于dataset的训练脚本
    - Fix Bug:
        - 修复logging模块重复输出日志的bug

- 1.0.0 (2021-04-27)
    - New Feature:
        - 按照模块功能, 重新调整模型的结构, 调整后的模块为: 
            - data: 数据预处理的各种功能函数
            - models: 自定义层, 常用模型
            - utils: 全局通用函数
            - metrics: 自定义衡量函数
            - pipeline: 流程控制函数
            - train: 自定义训练函数

## Version 0.9

- 0.9.0 (2020-12-13)
    - New Feature:
        - Add setting GPU function
        - Add FernSimpleTrainer
    - Style:
        - Update trainer's annotation

## Version 0.8

- 0.8.1 (2020-11-16)
    - Fix Bug:
        - Map function error without send a function
    - New Feature:
        - Support `parallel_apply` for FernDataFrame
        - Update FernCleaner to process data with parallel process
    - Test:
        - Add `test_parallel_apply` method to test `parallel_apply`

- 0.8.0 (2020-11-16)
    - New Feature:
        - Support `parallel_map` for FernSeries
    - Style:
        - Use Google annotation style in `data.py`

## Version 0.7

- 0.7.0 (2020-11-11)
    - New Feature:
        - Support multi inputs and outputs with dictionary format data
        - Label weight is no longer recommended, but dataset balancing is
    - Style:
        - Use a stub file to annotate the parameter types of the file `train.py`

## Version 0.6

- 0.6.4 (2020-11-10)
    - Bug Fix:
        - Fix data set length error when there is a multi input dictionary data

- 0.6.3 (2020-10-16)
    - Bug Fix:
        - ZeroDivisionError, while balancing data and num = 1

- 0.6.2 (2020-10-15)
    - Bug Fix:
        - Prefix code type error
        - Data and label column name error while training
        - Data type error while loading data from disk
        - Repeat prefix code

- 0.6.1 (2020-10-11)
    - Bug Fix:
        - No longer to delete character '<' and '>' for keeping special words like <SEP>

- 0.6.0 (2020-10-11)
    - New Feature:
        - Let FernDownloader download data from sql via SQLAlchemy
        - Treat a string like <ST> as a word not a string
        - Delete all unimportant words
    - Bug Fix:
        - KeyError while concatenate a Series to an array

## Version 0.5

- 0.5.1 (2020-07-27)
    - Bug fix:
        - Fix dataset_total length
    - Style:
        - Change loss function's `train` param to `with_label_weight`

- 0.5.0 (2020-07-26)
    - New Feature:
        - Add FernBalance class for dataset balance
        - Add gpu finder to find which gpu has the largest free memory
  
## Version 0.4

- 0.4.1 (2020-07-22)
    - New Feature:
        - Stop words use regular matching
        - The `#` comment rule is allowed in word library
    
- 0.4.0 (2020-07-21)
    - New Feature:
        - Add optimized TextCNN model as a built-in model

## Version 0.3

- 0.3.3 (2020-07-09)
    - New Feature:
        - Add predict function for FernModel
    
- 0.3.2 (2020-07-06)
    - Style:
        - Rename BaseTrainer -> FernTrainer
    - Bug Fix:
        - Fix trainer not fitting for multi output labels
 
- 0.3.1 (2020-07-02)
    - Bug Fix:
        - Fix no user words file exist issue
        - Fix cleaned label value should be a list issue
  
- 0.3.0 (2020-07-02)
    - New Feature:
        - using cut_func to split sequence to word list
        - Add save_function for every data precessing stage
        - Mark FernTransformer.output_shape as optional
        - Enable data frame default index
        - Add Sequence2Words class as cut_func generator
        - Use fern as default data frame 

## Version 0.2

- 0.2.0 (2020-05-24)
    - New Features:
        - Modify output shape from list into dict
        - Add label path for storing label data
        - Modify transformed label from array into dictionary
        - Update transformed label and data function
    - Style
        - Change variable name input_col to data_col
        - Change variable name output_col to label_col

## Version 0.1

- 0.1.4 (2020-05-20)
    - Style
        - Change the random state location where it's definded

- 0.1.3 (2020-05-20)
    - New Features
        - add random state for data splitter

- 0.1.2 (2020-05-07)
    - Bug fix
        - Fix training loop quick exit bug

- 0.1.1 (2020-05-03)
    - New features
        - Add github python publish action

- 0.1.0 (2020-05-03)
    - New features
        - Data preprocessing module: data downloader, data cleaner, data transformer and data splitter
        - Model template: model builder, saver, loader and structure printer
        - Model trainer
        - Custom layers: Conv1DPassMask, FlattenPassMask, DenseWithMask, AttentionLayer and ScaledDotProductAttention
        - Custom metrics: BinaryCategoricalAccuarcy
        - Other tools: logging and progress bar
    - Documentation changes
        - Add README 
        - Add README_ZH
