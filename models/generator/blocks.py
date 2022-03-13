import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    LeakyReLU,
)

from tensorflow.keras.layers import BatchNormalization

__INITIALIZER__ = None
Normalization = BatchNormalization
Activation = LeakyReLU


class ReflectionPad2D(tf.keras.layers.Layer):
    def __init__(self, paddings=(1, 1, 1, 1)):
        super(ReflectionPad2D, self).__init__()
        self.paddings = paddings

    def call(self, input):
        l, r, t, b = self.paddings

        return tf.pad(
            input, paddings=[[0, 0], [t, b], [l, r], [0, 0]], mode="REFLECT"
        )


class ConvBlock(Model):
    def __init__(
        self,
        outp: int,
        k: int,
        s: int,
        norm: bool = True,
        act: bool = True,
    ) -> None:
        super().__init__()
        layer = [
            ReflectionPad2D(),
            Conv2D(
                outp,
                k,
                s,
                kernel_initializer=__INITIALIZER__,
                use_bias=False,
            ),
        ]
        if norm:
            layer += [Normalization()]
        if act:
            layer += [Activation(0.2)]
        self.block = Sequential(layer)

    def call(self, x: Tensor, training: bool = None) -> Tensor:
        return self.block(x, training=training)


class ConvUpBlock(Model):
    def __init__(
        self,
        outp: int,
        k: int,
        s: int,
        norm: bool = True,
        act: bool = True,
    ) -> None:
        super().__init__()
        layer = [
            Conv2DTranspose(
                outp,
                k,
                s,
                padding="same",
                kernel_initializer=__INITIALIZER__,
                use_bias=False,
            ),
        ]
        if norm:
            layer += [Normalization()]
        if act:
            layer += [Activation(0.2)]
        self.block = Sequential(layer)

    def call(self, x: Tensor, training: bool = None) -> Tensor:
        return self.block(x, training=training)
