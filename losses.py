import tensorflow as tf
from tensorflow import Tensor


def binary_crossentropy(x: Tensor, y: Tensor) -> Tensor:
    return tf.reduce_mean(
        tf.keras.metrics.binary_crossentropy(y, x, from_logits=True, axis=0)
    )


def l1_loss(x: Tensor, y: Tensor) -> Tensor:
    return tf.reduce_mean(tf.abs(x - y))
