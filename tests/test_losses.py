import pytest
from losses import binary_crossentropy, l1_loss
from tests.conftest import build_tensor
import tensorflow as tf


def test_bce():
    logits = tf.zeros([100, 1])
    loss = binary_crossentropy(logits, logits)
    assert list(loss.shape) == []


@pytest.mark.parametrize("image_size", [128, 512])
def test_(image_size: int):
    image = build_tensor(1, image_size, 3)
    loss = l1_loss(image, image)
    assert loss == 0
