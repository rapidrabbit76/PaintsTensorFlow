from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import Tensor


def mask_gen(tensor_shape: List[int], X, zero_hint_size: int = None):
    # code from AlacGAN
    b, w, h, _ = tensor_shape
    zero_hint_size = b // 2 if zero_hint_size is None else zero_hint_size
    mask1 = np.concatenate(
        [
            (np.random.uniform(size=[1, w, h, 1]) >= X.rvs(1)[0]).astype(
                np.float32
            )
            for _ in range(b - zero_hint_size)
        ],
        0,
    )
    mask2 = np.concatenate(
        [
            np.zeros([1, w, h, 1]).astype(np.float32)
            for _ in range(zero_hint_size)
        ],
        0,
    )

    mask = np.concatenate([mask1, mask2], 0)
    return tf.convert_to_tensor(mask, dtype=tf.float32)


def logits_to_prob(x):
    return tf.reduce_mean(tf.nn.sigmoid(x), 0)
