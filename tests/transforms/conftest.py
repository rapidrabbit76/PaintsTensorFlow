from typing import Callable, Union

import pytest

from dataset import ColorizationTransforms, DraftTransforms, Mode


def build_transforms(
    args,
    transforms: Union[DraftTransforms, ColorizationTransforms],
    train: Mode,
) -> Callable:
    return transforms(
        image_size=args.image_size,
        draft_image_r=args.draft_image_r,
        train=train,
    )


@pytest.fixture(scope="session", params=[Mode.TRAIN, Mode.TEST])
def draft_transforms(request, args) -> Callable:
    return build_transforms(args, DraftTransforms, request.param)


@pytest.fixture(scope="session", params=[Mode.TRAIN, Mode.TEST])
def colorization_transforms(request, args) -> Callable:
    return build_transforms(args, ColorizationTransforms, request.param)
