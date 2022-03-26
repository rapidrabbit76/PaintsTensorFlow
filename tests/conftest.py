import os
import tempfile
import pytest
from easydict import EasyDict

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.keras import Model
from typing import Tuple
from models import Generator, Discriminator
import numpy as np


def build_image(image_size: int, channels: int) -> Tensor:
    image_shape = [image_size, image_size, channels]
    return tf.zeros(image_shape, dtype=tf.uint8)


def build_tensor(b, wh, c) -> Tensor:
    shape = [b, wh, wh, c]
    return tf.zeros(shape, dtype=tf.float32)


def build_image_batch(image_size) -> Tuple[Tensor, Tensor]:
    line = build_image(image_size, 1)
    color = build_image(image_size, 3)
    return (line, color)


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
def colorization_model(args) -> Model:
    return Generator(args.g_dim)


@pytest.fixture(scope="session")
def disc(args) -> Model:
    return Discriminator(args.d_dim)


@pytest.fixture(scope="session")
def draft_image_batch(args) -> np.ndarray:
    return build_image_batch(args.image_size // args.draft_image_r)


@pytest.fixture(scope="session")
def colorization_image_batch(args) -> np.ndarray:
    return build_image_batch(args.image_size)


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


@pytest.fixture(scope="session")
def save_dir():
    return tempfile.TemporaryDirectory()


@pytest.fixture(scope="session")
def draft_model_save_path(save_dir):
    save_dir = os.path.join(save_dir.name, "draft_model")
    return save_dir


@pytest.fixture(scope="session")
def colorization_model_save_path(save_dir):
    save_dir = os.path.join(save_dir.name, "colorization_model")
    return save_dir
