import pytest
from easydict import EasyDict

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.keras import Model
from typing import Tuple
from models import Generator, Discriminator
import numpy as np


@pytest.fixture(scope="session")
def args():
    return EasyDict(
        {
            "image_size": 512,
            "draft_image_r": 4,
            "batch_size": 1,
            "g_dim": 64,
            "d_dim": 64,
        }
    )


@pytest.fixture(scope="session")
def draft_model(args) -> Model:
    return Generator(args.g_dim)


@pytest.fixture(scope="session")
def gen(draft_model: Generator) -> Model:
    return draft_batch


@pytest.fixture(scope="session")
def colorization_model(args) -> Model:
    return Generator(args.g_dim)


@pytest.fixture(scope="session")
def disc(args) -> Model:
    return Discriminator(args.d_dim)


def build_image(image_size: int, channels: int) -> np.ndarray:
    image_shape = [image_size, image_size]
    if channels > 1:
        image_shape += [channels]
    return np.zeros(image_shape, dtype=np.uint8)


def build_tensor(b, wh, c) -> Tensor:
    w, h = wh
    shape = [b, w, h, c]
    return tf.zeros(shape, dtype=tf.float32)


@pytest.fixture(scope="session")
def line_image(args) -> np.ndarray:
    return build_image(args.image_sze, 1)


@pytest.fixture(scope="session")
def color_image(args) -> np.ndarray:
    return build_image(args.image_sze, 3)


@pytest.fixture(scope="session")
def draft_batch(args) -> Tuple[Tensor, Tensor, Tensor]:
    b = args.batch_size
    wh = args.image_size // args.draft_image_r
    line = build_tensor(b, wh, 1)
    color = build_tensor(b, wh, 3)
    hint = tf.zeros_like(color)
    return (line, hint, color)


@pytest.fixture(scope="session")
def colorization_batch(
    args, draft_batch
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    b = args.batch_size
    wh = args.image_size
    line = build_tensor(b, wh, 1)
    color = build_tensor(b, wh, 3)
    line_draft, hint, _ = draft_batch
    return (line, line_draft, hint, color)
