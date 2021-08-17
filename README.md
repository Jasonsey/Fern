# FERN

[![Pypi](https://github.com/Jasonsey/Fern/actions/workflows/pypi.yml/badge.svg)](https://github.com/Jasonsey/Fern/actions/workflows/pypi.yml)

Fern用于NLP的模型开发结构控制。通过它可以控制文本预处理、模型搭建、训练器：

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
   $ pip install -e git+https://github.com/Jasonsey/Fern.git
   ```

## 使用教程

建议查看源码中函数的使用说明

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
