# Fern
# 
# @Author: Jason, Lin
# @Email : Jason.M.Lin@outlook.com
# @Time  : 2022/2/9 9:15 上午
#
# =============================================================================
"""特定模型直接训练调度代码，同时也是为了解耦model&train代码的嵌套import。几个模块之间的关系如下

Estimator:
    Data Pipeline
    Model
    Trainer
"""
from typing import *
from fern.train import Word2VecTrainer
from fern.models import Word2Vec
from fern.pipeline import Word2VecDataPipeline


def word2vec_estimator(
        data: List[str],
        batch_size: int,
        zh_segmentation: bool,
        model_home: str,
        ckpt_home: str,
        log_home: str,
        epoch: int,
        embedding_dim: int = 128,
        version: int = 1,
        opt: str = 'adam',
        lr: float = 1.0e-3,
        lower: bool = True,
        win_size: int = 2,
        random_seed: int = 42,
        verbose: int = 1) -> Tuple[Word2Vec, List[str], float]:
    """调用对应的训练器训练模型

    Examples:
        >>> # 训练
        >>> import tensorflow as tf
        >>> from fern.estimate import word2vec_estimator
        >>> path_to_file = tf.keras.utils.get_file('shakespeare.txt',
        ...     'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        >>> with open(path_to_file, 'r') as f:
        ...     data = []
        ...     for line in f:
        ...         if line.strip():
        ...             data.append(line.strip())
        >>> model, vocab, loss = word2vec_estimator(
        ...     data=data[:100],
        ...     batch_size=128,
        ...     zh_segmentation=True,
        ...     model_home='/tmp/Fern/model',
        ...     ckpt_home='/tmp/Fern/ckpt',
        ...     log_home='/tmp/Fern/log',
        ...     epoch=200,
        ...     embedding_dim=8,
        ...     version=1,
        ...     opt='adam',
        ...     lr=1.0e-3,
        ...     lower=True,
        ...     win_size=2,
        ...     random_seed=42,
        ...     verbose=1)
        >>> print(loss)
        >>> # 获取词向量
        >>> model.target_embedding.variables[0].numpy()

    Args:
        data: 训练数据，每个元素都是一个句子
        batch_size: 每批次大小
        zh_segmentation: 是否做中文分词, 否则中文按字分割
        model_home: 模型保存根目录
        ckpt_home: 模型检查点目录
        log_home: 日志检查点目录
        epoch: 循环训练多少次
        embedding_dim: word2vec的词向量大小
        version: 模型的版本号
        opt: 优化器
        lr: 学习率
        lower: 是否忽略大小写
        win_size: 生成(target, context)的窗口大小
        random_seed: 随机种子大小
        verbose: 0: 不打印日志；1打印epoch日志；2: 打印batch日志

    Returns:
        (model, vocab, train_loss): 模型，词库，训练损失
    """
    pipeline = Word2VecDataPipeline(data=data, batch_size=batch_size, zh_segmentation=zh_segmentation, lower=lower,
                                    win_size=win_size, random_seed=random_seed)
    train_ds = pipeline.train_ds
    val_ds = pipeline.val_ds

    model = Word2Vec(vocab_size=pipeline.vocab_size, embedding_dim=embedding_dim)

    trainer = Word2VecTrainer(model=model, model_home=model_home, ckpt_home=ckpt_home, log_home=log_home,
                              train_ds=train_ds, val_ds=val_ds, opt=opt, lr=lr, version=version)
    train_loss = trainer.train(epoch=epoch, verbose=verbose)
    return model, list(pipeline.word2id.keys()), train_loss
