import os
from typing import Dict, List, Tuple, Union

import numpy as np
import opt
import scipy.stats as stats
import tensorflow as tf
import tensorflow.keras.optimizers as optim
import wandb
from dataset import ColorizationTransforms, Mode, build_dataset
from models import Generator
from tensorflow import Tensor
from tqdm import tqdm
import losses
from trainer.draft import log_step

global global_step
global_step = 0

# Line, Line_draft , Hint, Color
batch_type = Tuple[Tensor, Tensor, Tensor, Tensor]


def colorization_train(args, wandb_run):
    assert os.path.exists(
        args.draft_weights_path
    ), f"{args.draft_weights_path} not exists"

    global logger, X
    logger = wandb_run
    os.makedirs(os.path.join(logger.dir, "image"), exist_ok=True)

    ##################### init values ##########################
    X = stats.truncnorm(
        (0 - args.mu) / args.sigma,
        (1 - args.mu) / args.sigma,
        loc=args.mu,
        scale=args.sigma,
    )
    ########################## dataset ##########################
    train_transforms = ColorizationTransforms(
        args.image_size, args.draft_image_r, train=Mode.TRAIN
    )
    test_transforms = ColorizationTransforms(
        args.image_size, args.draft_image_r, train=Mode.TEST
    )

    train_dataset = build_dataset(
        os.path.join(args.root_dir, "train"), train_transforms, args.batch_size
    )

    test_dataset = build_dataset(
        os.path.join(args.root_dir, "test"), test_transforms, 8
    )

    test_batch = next(iter(test_dataset))
    line, line_draft, hint, color = test_batch
    mask = opt.mask_gen(list(hint.shape), X, 0)
    hint = hint * mask
    test_batch = (line, line_draft, hint, color)

    ########################## model ##########################
    draft_image_size = args.image_size // args.draft_image_r
    draft_gen = Generator(args.g_dim)
    draft_gen(
        tf.zeros([1, draft_image_size, draft_image_size, 1]),
        tf.zeros([1, draft_image_size, draft_image_size, 3]),
        training=False,
    )
    draft_gen.load_weights(args.draft_weights_path)
    draft_gen.trainable = False

    gen = Generator(args.g_dim)

    ########################## optim ##########################
    total_step = len(train_dataset) * args.epochs

    gen_optim_lr_schedule = optim.schedules.ExponentialDecay(
        args.lr,
        decay_steps=int(total_step * args.decay_steps_rate),
        decay_rate=args.decay_rate,
        staircase=True,
    )

    gen_optim = optim.Adam(
        gen_optim_lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2
    )

    ######################## training ##########################
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch} start")
        training_loop(train_dataset, test_batch, draft_gen, gen, gen_optim)
        gen.save_weights(os.path.join(logger.dir, "gen.ckpt.h5"))

    #################### artifacts loging ######################
    artifacts_path = os.path.join(logger.dir, f"{args.mode}.h5")
    gen.save(artifacts_path)

    if args.upload_artifacts:
        gen_artifacts = wandb.Artifact(args.mode, type="model")
        gen_artifacts.add_file(artifacts_path, "weight")
        logger.log_artifact(gen_artifacts)


def training_loop(train_dataset, test_batch, draft_gen, gen, gen_optim):
    global global_step
    last_batch_idx = len(train_dataset) - 1
    pbar = tqdm(train_dataset)
    for batch_idx, batch in enumerate(pbar):
        global_step += 1
        training_info = training_step(draft_gen, gen, gen_optim, batch)

        if batch_idx % 5 == 0:
            log_dict = {"loss": training_info["l1_loss"]}
            logger.log(log_dict)
            pbar.set_description_str(
                (
                    f"[GS:{global_step} "
                    f"[l1:{log_dict['loss'].numpy().item(): 0.2f}]"
                ),
            )

        if batch_idx % 100 == 0 or batch_idx == last_batch_idx:
            test_info = test_step(test_batch, draft_gen, gen)
            size = test_info["line"].shape[1]
            train_images = [
                training_info["line"],
                tf.image.resize(training_info["hint"], (size, size)),
                training_info["draft"],
                training_info["color"],
                training_info["_color"],
            ]
            test_images = [
                test_info["line"],
                tf.image.resize(test_info["hint"], (size, size)),
                test_info["draft_w_hint"],
                test_info["draft_wo_hint"],
                test_info["_color_w_hint"],
                test_info["_color_wo_hint"],
                test_info["color"],
            ]
            log_step(logger, train_images, test_images)


@tf.function
def training_step(
    draft_gen: Generator,
    gen: Generator,
    gen_optim,
    batch: Tuple[Tensor, Tensor, Tensor],
) -> Dict[str, Tensor]:
    line, line_draft, hint, color = batch
    batch_size = line.shape[0]
    mask = opt.mask_gen(list(hint.shape), X, batch_size // 2)
    hint = hint * mask
    image_size = line.shape[1]

    draft = build_draft(draft_gen, line_draft, hint, image_size)

    with tf.GradientTape() as tape:
        _color = gen({"line": line, "hint": draft, "training": True})
        loss = losses.l1_loss(_color, color)

    grad = tape.gradient(loss, gen.trainable_variables)
    gen_optim.apply_gradients(zip(grad, gen.trainable_variables))

    return {
        # image
        "line": line,
        "hint": hint,
        "color": color,
        "draft": draft,
        "_color": _color,
        # scaler
        "l1_loss": loss,
    }


@tf.function
def build_draft(
    draft_gen: Generator,
    line_draft: Tensor,
    hint: Tensor,
    size: int,
) -> Tensor:
    draft = draft_gen({"line": line_draft, "hint": hint, "training": False})
    draft = tf.image.resize(
        draft, (size, size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return draft


@tf.function
def test_step(
    test_batch: batch_type, draft_gen: Generator, gen: Generator
) -> List[Tensor]:
    line, line_draft, hint, color = test_batch
    zero_hint = tf.zeros_like(hint)
    image_size = line.shape[1]

    draft_w_hint = build_draft(draft_gen, line_draft, hint, image_size)
    draft_wo_hint = build_draft(draft_gen, line_draft, zero_hint, image_size)

    _color_w_hint = gen(
        {"line": line, "hint": draft_w_hint, "training": False},
    )
    _color_wo_hint = gen(
        {"line": line, "hint": draft_wo_hint, "training": False}
    )

    hint = tf.image.resize(hint, (image_size, image_size))
    return {
        # images
        "line": line,
        "hint": hint,
        "color": color,
        "draft_w_hint": draft_w_hint,
        "draft_wo_hint": draft_wo_hint,
        "_color_w_hint": _color_w_hint,
        "_color_wo_hint": _color_wo_hint,
    }
