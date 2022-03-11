from typing import Callable, Tuple
import numpy as np
from tensorflow import Tensor


def test_draft_transforms(
    draft_transforms: Callable,
    draft_image_batch: Tuple[Tensor, Tensor],
    draft_batch: Tuple[Tensor, Tensor, Tensor],
):
    check_transfoms_shape(draft_transforms, draft_image_batch, draft_batch)


def test_colorization_transforms(
    colorization_transforms: Callable,
    colorization_image_batch: Tuple[Tensor, Tensor],
    colorization_batch: Tuple[Tensor, Tensor, Tensor],
):
    check_transfoms_shape(
        colorization_transforms, colorization_image_batch, colorization_batch
    )


def check_transfoms_shape(transforms, image_batch, target_batch):
    line, color = image_batch
    batch = transforms.preprocessing(line, color)

    batch = list(batch)
    target_batch = list(target_batch)

    batch = [list(bat.shape) for bat in batch]
    target_batch = [list(db[0].shape) for db in target_batch]

    for batch_shape, target_batch_shape in zip(batch, target_batch):
        assert batch_shape == target_batch_shape
