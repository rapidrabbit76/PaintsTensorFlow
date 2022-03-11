from typing import *
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D

from .blocks import ConvBlock, ConvUpBlock


class Generator(Model):
    def __init__(
        self,
        dim: int,
        encoding_layers: List[Union[float, int]] = [
            0.5,
            1,
            1,
            2,
            2,
            4,
            4,
            8,
            8,
        ],
    ):
        super().__init__()
        encoding_layers = [int(dim * f) for f in encoding_layers]
        decoding_layers = encoding_layers[::-1]
        (
            self.e0,
            self.e1,
            self.e2,
            self.e3,
            self.e4,
            self.e5,
            self.e6,
            self.e7,
            self.e8,
        ) = self._make_encoding_layers(encoding_layers)
        (
            self.d8,
            self.d7,
            self.d6,
            self.d5,
            self.d4,
            self.d3,
            self.d2,
            self.d1,
        ) = self._make_decoding_layers(decoding_layers[1:])

        self.last = Conv2D(
            3, 3, 1, padding="same", use_bias=False, activation=tf.nn.tanh
        )

    def _make_encoding_layers(self, layers: List[int]) -> List[Model]:
        blocks = []
        for i, dim in enumerate(layers):
            blocks += [
                ConvBlock(
                    dim,
                    3 if i % 2 == 0 else 4,
                    1 if i % 2 == 0 else 2,
                )
            ]
        return blocks

    def _make_decoding_layers(self, layers: List[int]) -> List[Model]:
        blocks = []
        for i, dim in enumerate(layers):
            Block = ConvUpBlock if i % 2 == 0 else ConvBlock
            blocks += [
                Block(
                    dim,
                    4 if i % 2 == 0 else 3,
                    2 if i % 2 == 0 else 1,
                )
            ]
        return blocks

    def call(self, line: Tensor, hint: Tensor, training=None) -> Tensor:
        x = tf.concat([line, hint], axis=-1)
        e0 = self.e0(x, training)
        e1 = self.e1(e0, training)
        e2 = self.e2(e1, training)
        e3 = self.e3(e2, training)
        e4 = self.e4(e3, training)
        e5 = self.e5(e4, training)
        e6 = self.e6(e5, training)
        e7 = self.e7(e6, training)
        e8 = self.e8(e7, training)

        d8 = self.d8(tf.concat([e7, e8], axis=-1), training)
        x = self.d7(d8, training)
        d6 = self.d6(tf.concat([d8, x], axis=-1), training)
        x = self.d5(d6, training)
        d4 = self.d4(tf.concat([d6, x], axis=-1), training)
        x = self.d3(d4, training)
        d2 = self.d2(tf.concat([d4, x], axis=-1), training)
        x = self.d1(d2, training)
        return self.last(tf.concat([e0, x], axis=-1))
