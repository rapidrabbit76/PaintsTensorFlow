from typing import *
from typing import Callable, List, Tuple

import albumentations as A
import cv2
import tensorflow as tf
from tensorflow import Tensor

from dataset import Mode


class DraftTransforms:
    def __init__(
        self,
        image_size: int,
        draft_image_r: float,
        train: Mode,
    ):
        super().__init__()
        image_size = image_size // draft_image_r
        self.both_transforms = A.Compose(
            [
                A.Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
                A.HorizontalFlip(p=0.5 if train is Mode.TRAIN else 0),
            ],
            additional_targets={"image0": "image"},
        )

    @classmethod
    def cast(cls, image: Tensor) -> Tensor:
        image = tf.cast(image, tf.float32) / 255
        image = (image - 0.5) / 0.5
        return image

    @classmethod
    def image_load(cls, path: str, channels: int = 3) -> Tensor:
        encoded_image = tf.io.read_file(path)
        image = tf.io.decode_image(encoded_image, channels)
        return image

    def b_transforms(self, line, color):
        data = {
            "image": line,
            "image0": color,
        }
        augmentations = self.both_transforms(**data)
        line = augmentations["image"]
        color = augmentations["image0"]
        return [line, color]

    def __call__(
        self, line_path: str, color_path: str
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        line = self.image_load(line_path, 1)
        color = self.image_load(color_path, 3)
        return self.preprocessing(line, color)

    def preprocessing(
        self, line: Tensor, color: Tensor
    ) -> Tuple[Tensor, Tensor]:

        line, color = tf.numpy_function(
            func=self.b_transforms,
            inp=[line, color],
            Tout=[tf.uint8, tf.uint8],
        )
        line = self.cast(line)
        color = self.cast(color)
        hint = tf.identity(color)

        return line, hint, color


class ColorizationTransforms(DraftTransforms):
    def __init__(
        self,
        image_size: int,
        draft_image_r: float,
        train: Mode,
    ):
        draft_image = image_size // draft_image_r
        self.both_transforms = A.Compose(
            [
                A.Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
                A.HorizontalFlip(p=0.5 if train is Mode.TRAIN else 0),
            ],
            additional_targets={"image0": "image"},
        )
        self.draft_transforms = A.Compose(
            [A.Resize(draft_image, draft_image, interpolation=cv2.INTER_AREA)],
            additional_targets={"image0": "image"},
        )

    def b_transforms(self, line, color):
        data = {
            "image": line,
            "image0": color,
        }

        augmentations = self.draft_transforms(**data)
        line_draft = augmentations["image"]
        hint = augmentations["image0"]
        return [line, line_draft, hint, color]

    def preprocessing(
        self, line: Tensor, color: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        line, line_draft, hint, color = tf.numpy_function(
            func=self.b_transforms,
            inp=[line, color],
            Tout=[tf.uint8, tf.uint8, tf.uint8, tf.uint8],
        )
        line = self.cast(line)
        line_draft = self.cast(line_draft)
        hint = self.cast(hint)
        color = self.cast(color)

        return line, line_draft, hint, color
