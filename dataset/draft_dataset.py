from typing import *
import os
import tensorflow as tf
from tensorflow import Tensor
import albumentations as A
from glob import glob

import cv2


class DraftDataset:
    def __init__(
        self,
        image_size: int,
        train: bool,
        draft_image_r: float = 4,
    ):
        super().__init__()
        draft_image = image_size // draft_image_r
        self.both_transforms = A.Compose(
            [
                A.Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
                A.RandomResizedCrop(draft_image, draft_image, p=1)
                if train
                else A.Resize(draft_image, draft_image),
                A.HorizontalFlip(p=0.5 if train else 0),
            ],
            additional_targets={"image0": "image"},
        )
        self.line_transforms = A.Compose(
            [
                A.Normalize(),
            ]
        )
        self.color_transforms = A.Compose([])

    @classmethod
    def cast(cls, image: Tensor) -> Tensor:
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
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
    ) -> Tuple[Tensor, Tensor]:
        line = self.image_load(line_path, 1)
        color = self.image_load(color_path, 3)

        line, color = tf.numpy_function(
            func=self.b_transforms, inp=[line, color], Tout=[tf.uint8, tf.uint8]
        )
        line = self.cast(line)
        color = self.cast(color)
        hint = tf.identity(color)

        return line, hint, color


def build_dataset(
    root_dir: str,
    image_size: int,
    batch_size: int,
    train: bool,
):
    line_paths = sorted(glob(os.path.join(root_dir, "line", "*")))
    color_paths = sorted(glob(os.path.join(root_dir, "image", "*")))
    assert len(line_paths) == len(color_paths), "line, color not paired"

    transforms = DraftDataset(image_size, train)

    datasets = tf.data.Dataset.from_tensor_slices((line_paths, color_paths))
    datasets = datasets.map(lambda x, y: transforms(x, y)).batch(batch_size)
    return datasets


if __name__ == "__main__":
    ds = build_dataset("DATASET/train", 512, 32, False)

    for line, hint, color in ds:
        print(line.shape, hint.shape, color.shape)
