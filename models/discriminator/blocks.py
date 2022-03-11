import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    Conv2DTranspose,
    LeakyReLU,
)
from tensorflow_addons.layers import InstanceNormalization


__INITIALIZER__ = None
Normalization = InstanceNormalization
Activation = LeakyReLU


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
            Conv2D(
                outp,
                k,
                s,
                padding="same",
                kernel_initializer=__INITIALIZER__,
                use_bias=False,
            )
        ]
        if norm:
            layer += [Normalization()]
        if act:
            layer += [Activation(0.2)]
        self.block = Sequential(layer)

    def call(self, x: Tensor, training: bool = None) -> Tensor:
        return self.block(x, training=training)
