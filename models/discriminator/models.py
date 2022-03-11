from typing import *
from tensorflow import Tensor
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten

from .blocks import ConvBlock


class Discriminator(Model):
    def __init__(
        self,
        dim: int,
        layer_dims: List[Union[float, int]] = [0.5, 0.5, 1, 1, 2, 2, 4],
    ):
        super().__init__()
        dims = [int(dim * f) for f in layer_dims]
        layers = [self._make_layers(dim, i) for i, dim in enumerate(dims)]
        layers += [ConvBlock(dim * 4, 4, 2, False)]
        self.layer = Sequential(layers)
        self.head = Sequential([Flatten(), Dense(1)])

    def _make_layers(self, dim: int, i: int):
        k = 4 if i % 2 == 0 else 3
        s = 2 if i % 2 == 0 else 1
        return ConvBlock(dim, k, s)

    def call(self, x: Tensor, training=None) -> Tensor:
        x = self.layer(x, training)
        return self.head(x, training)
