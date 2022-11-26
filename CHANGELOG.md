## [Unreleased]

## [1.4.0] - 2022-11-26
### Added
- 添加繁体-简体互相转换功能 #11
- 添加中文词列表合并字符串功能 #13

## [1.3.0] - 2022-11-26
### Added
- 添加macroF1评估函数 #12

## [1.2.2] - 2022-11-26
### Fixed
- 修复The truth value of a DataFrame is ambiguous #9

## [1.2.1] - 2021-11-22
### Fixed
- 确保代码兼容python3.6

## [1.2.0] - 2021-10-18
### Added
- 添加BERT预训练模型, 基于tensorflow hub
- 添加针对tensor和ragged tensor的通用map函数
- 实现一个函数导出所有的bert配置

## [1.1.4] - 2021-09-24
### Fixed
- 修复依赖文件
- 修复log的继承关系
- 修复语法上的问题
    
## [1.1.3] - 2021-09-24
### Fixed
- 修复缺少sklearn依赖的问题

## [1.1.2] - 2021-08-17
### Added
- 添加常用的自定义loss
- 添加Logging的上下文管理模块, 控制节点
- 添加字符串转驼峰脚本
- 默认创建的logging添加stream logging功能

## [1.1.1] - 2021-07-18
### Fixed
- 修复Optional的错误
- 默认的logger不允许从父类中继承handler
- 修复分割数据集时, 大于1浮点数问题
- 更新logging的默认名字
- 更新发布流程: 打上标签即发布

## [1.1.0] - 2021-07-14
### Added
- 添加Bert token方法
- 添加yaml文件读取工具
- 添加keras模型快速编译函数
- 添加keras模型基于dataset的训练脚本

### Fixed
- 修复logging模块重复输出日志的bug

## [1.0.0] - 2021-04-27
### Added
- 按照模块功能, 重新调整模型的结构, 调整后的模块为: 
    - data: 数据预处理的各种功能函数
    - models: 自定义层, 常用模型
    - utils: 全局通用函数
    - metrics: 自定义衡量函数
    - pipeline: 流程控制函数
    - train: 自定义训练函数

## [0.9.0] - 2020-12-13
### Added
- Add setting GPU function
- Add FernSimpleTrainer

### Changed
- Update trainer's annotation

## [0.8.1] - 2020-11-16
### Added
- Support `parallel_apply` for FernDataFrame
- Update FernCleaner to process data with parallel process
- Add `test_parallel_apply` method to test `parallel_apply`

### Fixed
- Map function error without send a function

## [0.8.0] - 2020-11-16
### Added
- Support `parallel_map` for FernSeries
- Use Google annotation style in `data.py`

## [0.7.0] - 2020-11-11
### Added
- Support multi inputs and outputs with dictionary format data
- Label weight is no longer recommended, but dataset balancing is
- Use a stub file to annotate the parameter types of the file `train.py`


## [0.6.4] - 2020-11-10
### Fixed
- Fix data set length error when there is a multi input dictionary data

## [0.6.3] - 2020-10-16
### Added
- ZeroDivisionError, while balancing data and num = 1

## [0.6.2] - 2020-10-15
### Fixed
- Prefix code type error
- Data and label column name error while training
- Data type error while loading data from disk
- Repeat prefix code

## [0.6.1] - 2020-10-11
### Fixed
- No longer to delete character '<' and '>' for keeping special words like <SEP>

## [0.6.0] - 2020-10-11
### Added
- Let FernDownloader download data from sql via SQLAlchemy
- Treat a string like <ST> as a word not a string

### Removed
- Delete all unimportant words

### Fixed
- KeyError while concatenate a Series to an array

## [0.5.1] - 2020-07-27
### Fixed
- Fix dataset_total length

### Changed
- Change loss function's `train` param to `with_label_weight`

## [0.5.0] - 2020-07-26
### Added
- Add FernBalance class for dataset balance
- Add gpu finder to find which gpu has the largest free memory
  
## [0.4.1] - 2020-07-22
### Added
- Stop words use regular matching
- The `#` comment rule is allowed in word library
    
## [0.4.0] - 2020-07-21
### Added
- Add optimized TextCNN model as a built-in model

## [0.3.3] - 2020-07-09
### Added
- Add predict function for FernModel
    
## [0.3.2] - 2020-07-06
### Changed
- Rename BaseTrainer -> FernTrainer

### Fixed
- Fix trainer not fitting for multi output labels
 
## [0.3.1] - 2020-07-02
### Fixed
- Fix no user words file exist issue
- Fix cleaned label value should be a list issue
  
## [0.3.0] - 2020-07-02
### Added
- using cut_func to split sequence to word list
- Add save_function for every data precessing stage
- Mark FernTransformer.output_shape as optional
- Enable data frame default index
- Add Sequence2Words class as cut_func generator
- Use fern as default data frame 

## [0.2.0] - 2020-05-24
### Added
- Add label path for storing label data

### Changed
- Modify output shape from list into dict
- Modify transformed label from array into dictionary
- Update transformed label and data function
- Change variable name input_col to data_col
- Change variable name output_col to label_col

## [0.1.4] - 2020-05-20
### Changed
- Change the random state location where it's defined

## [0.1.3] - 2020-05-20
### Added
- add random state for data splitter

## [0.1.2] - 2020-05-07
### Fixed
- Fix training loop quick exit bug


## [0.1.1] - 2020-05-03
### Added
- Add github python publish action

## [0.1.0] - 2020-05-03
### Added
- Data preprocessing module: data downloader, data cleaner, data transformer and data splitter
- Model template: model builder, saver, loader and structure printer
- Model trainer
- Custom layers: Conv1DPassMask, FlattenPassMask, DenseWithMask, AttentionLayer and ScaledDotProductAttention
- Custom metrics: BinaryCategoricalAccuarcy
- Other tools: logging and progress bar
- Add README 
- Add README_ZH


[Unreleased]: https://github.com/Jasonsey/Fern/compare/1.4.0...HEAD
[1.4.0]: https://github.com/Jasonsey/Fern/compare/1.3.0...1.4.0
[1.3.0]: https://github.com/Jasonsey/Fern/compare/1.2.2...1.3.0
[1.2.2]: https://github.com/Jasonsey/Fern/compare/1.2.1...1.2.2
[1.2.1]: https://github.com/Jasonsey/Fern/compare/1.2.0...1.2.1
[1.2.0]: https://github.com/Jasonsey/Fern/compare/1.1.4...1.2.0
[1.1.4]: https://github.com/Jasonsey/Fern/compare/1.1.3...1.1.4
[1.1.3]: https://github.com/Jasonsey/Fern/compare/1.1.2...1.1.3
[1.1.2]: https://github.com/Jasonsey/Fern/compare/1.1.1...1.1.2
[1.1.1]: https://github.com/Jasonsey/Fern/compare/1.1.0...1.1.1
[1.1.0]: https://github.com/Jasonsey/Fern/compare/1.0.0...1.1.0
[1.0.0]: https://github.com/Jasonsey/Fern/compare/0.9.0...1.0.0
[0.9.0]: https://github.com/Jasonsey/Fern/compare/0.8.1...0.9.0
[0.8.1]: https://github.com/Jasonsey/Fern/compare/0.8.0...0.8.1
[0.8.0]: https://github.com/Jasonsey/Fern/compare/0.7.0...0.8.0
[0.7.0]: https://github.com/Jasonsey/Fern/compare/0.6.4...0.7.0
[0.6.4]: https://github.com/Jasonsey/Fern/compare/0.6.3...0.6.4
[0.6.3]: https://github.com/Jasonsey/Fern/compare/0.6.2...0.6.3
[0.6.2]: https://github.com/Jasonsey/Fern/compare/0.6.1...0.6.2
[0.6.1]: https://github.com/Jasonsey/Fern/compare/0.6.0...0.6.1
[0.6.0]: https://github.com/Jasonsey/Fern/compare/0.5.1...0.6.0
[0.5.1]: https://github.com/Jasonsey/Fern/compare/0.5.0...0.5.1
[0.5.0]: https://github.com/Jasonsey/Fern/compare/0.4.1...0.5.0
[0.4.1]: https://github.com/Jasonsey/Fern/compare/0.4.0...0.4.1
[0.4.0]: https://github.com/Jasonsey/Fern/compare/0.3.3...0.4.0
[0.3.3]: https://github.com/Jasonsey/Fern/compare/0.3.2...0.3.3
[0.3.2]: https://github.com/Jasonsey/Fern/compare/0.3.1...0.3.2
[0.3.1]: https://github.com/Jasonsey/Fern/compare/0.3.0...0.3.1
[0.3.0]: https://github.com/Jasonsey/Fern/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/Jasonsey/Fern/compare/0.1.4...0.2.0
[0.1.4]: https://github.com/Jasonsey/Fern/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/Jasonsey/Fern/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/Jasonsey/Fern/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/Jasonsey/Fern/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/Jasonsey/Fern/releases/tag/0.1.0
