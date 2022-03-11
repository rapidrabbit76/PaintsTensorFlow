import os
from glob import glob
from typing import Callable, List, Tuple

import tensorflow as tf


from enum import Enum, Flag


class Mode(Enum):
    TRAIN: str = "train"
    TEST: str = "test"


def build_dataset(
    root_dir: str,
    transforms: Callable,
    batch_size: int,
) -> tf.data.Dataset:
    line_paths, color_paths = _load_paths(root_dir)
    datasets = tf.data.Dataset.from_tensor_slices((line_paths, color_paths))
    datasets = datasets.map(lambda x, y: transforms(x, y)).batch(batch_size)
    return datasets


def _load_paths(root_dir) -> Tuple[List[str], List[str]]:
    line_paths = sorted(glob(os.path.join(root_dir, "line", "*")))
    color_paths = sorted(glob(os.path.join(root_dir, "image", "*")))
    assert len(line_paths) == len(color_paths), "line, color not paired"
    return line_paths, color_paths
