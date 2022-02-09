# Fern
# 
# @Author: Jason, Lin
# @Email : Jason.M.Lin@outlook.com
# @Time  : 2022/1/22 7:34 下午
#
# =============================================================================
"""test_models.py"""
import tensorflow as tf
from fern.estimate import word2vec_estimator


def test_word2vec():
    path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                           'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    with open(path_to_file, 'r') as f:
        data = []
        for line in f:
            if line.strip():
                data.append(line.strip())
    model, vocab, loss = word2vec_estimator(
        data=data[:100],
        batch_size=128,
        zh_segmentation=True,
        model_home='/tmp/Fern/model',
        ckpt_home='/tmp/Fern/ckpt',
        log_home='/tmp/Fern/log',
        epoch=200,
        embedding_dim=8,
        version=1,
        opt='adam',
        lr=1.0e-3,
        lower=True,
        win_size=2,
        random_seed=42,
        verbose=1)
    print(loss)
