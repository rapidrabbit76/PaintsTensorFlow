import os
from typing import Dict, List, Tuple, Union

import numpy as np
import opt
import scipy.stats as stats
import tensorflow as tf
import tensorflow.keras.optimizers as optim
import wandb
from dataset import DraftTransforms, Mode, build_dataset
from models import Discriminator, Generator
from tensorflow import Tensor
from tqdm import tqdm
import losses

global global_step
global_step = 0

# Line , Hint, Color
batch_type = Tuple[Tensor, Tensor, Tensor]


def draft_train(args, wandb_run):
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
    train_transforms = DraftTransforms(
        args.image_size, args.draft_image_r, train=Mode.TRAIN
    )
    test_transforms = DraftTransforms(
        args.image_size, args.draft_image_r, train=Mode.TEST
    )

    train_dataset = build_dataset(
        os.path.join(args.root_dir, "train"), train_transforms, args.batch_size
    )

    test_dataset = build_dataset(
        os.path.join(args.root_dir, "test"), test_transforms, 8
    )

    test_batch = next(iter(test_dataset))
    line, hint, color = test_batch
    mask = opt.mask_gen(list(hint.shape), X, 0)
    hint = hint * mask
    test_batch = (line, hint, color)

    ########################## model ##########################
    gen = Generator(args.g_dim)
    disc = Discriminator(args.d_dim)

    ########################## optim ##########################
    total_step = len(train_dataset) * args.epochs

    gen_optim_lr_schedule = optim.schedules.ExponentialDecay(
        args.lr,
        decay_steps=int(total_step * args.decay_steps_rate),
        decay_rate=args.decay_rate,
        staircase=True,
    )
    disc_optim_lr_schedule = optim.schedules.ExponentialDecay(
        args.lr,
        decay_steps=int(total_step * args.decay_steps_rate),
        decay_rate=args.decay_rate,
        staircase=True,
    )

    gen_optim = optim.Adam(
        gen_optim_lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2
    )
    disc_optim = optim.Adam(
        disc_optim_lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2
    )

    ######################## training ##########################
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch} start")
        training_loop(
            train_dataset, test_batch, gen, disc, gen_optim, disc_optim
        )

    #################### artifacts loging ######################
    artifacts_path = os.path.join(logger.dir, f"{args.mode}.h5")
    gen.save(artifacts_path)

    if args.upload_artifacts:
        gen_artifacts = wandb.Artifact(args.mode, type="model")
        gen_artifacts.add_file(artifacts_path, "weight")
        logger.log_artifact(gen_artifacts)


def training_loop(train_dataset, test_batch, gen, disc, gen_optim, disc_optim):
    global global_step
    last_batch_idx = len(train_dataset) - 1
    pbar = tqdm(train_dataset)
    for batch_idx, batch in enumerate(pbar):
        global_step += 1
        training_info = training_step(gen, disc, gen_optim, disc_optim, batch)

        if batch_idx % 5 == 0:
            log_dict = {
                "gan/real_prob": opt.logits_to_prob(training_info["logits"]),
                "gan/fake_prob": opt.logits_to_prob(training_info["_logits"]),
                "gen/adv_loss": training_info["adv_loss"],
                "gen/l1_loss": training_info["l1_loss"],
                "gen/loss": training_info["gen_loss"],
                "disc/real_loss": training_info["real_loss"],
                "disc/fake_loss": training_info["fake_loss"],
                "disc/loss": training_info["disc_loss"],
            }
            logger.log(log_dict)
            pbar.set_description_str(
                (
                    f"[GS:{global_step} "
                    f"[rp:{log_dict['gan/real_prob'].numpy().item(): 0.2f}] "
                    f"[fp:{log_dict['gan/fake_prob'].numpy().item(): 0.2f}] "
                    f"[l1:{log_dict['gen/l1_loss'].numpy().item(): 0.2f}]"
                ),
            )

        if batch_idx % 100 == 0 or batch_idx == last_batch_idx:
            test_info = test_step(test_batch, gen)
            train_images = [
                training_info["line"],
                training_info["hint"],
                training_info["color"],
                training_info["_color"],
            ]
            test_images = [
                test_info["line"],
                test_info["hint"],
                test_info["_color_w_hint"],
                test_info["_color_wo_hint"],
                test_info["color"],
            ]
            log_step(logger, train_images, test_images)


@tf.function
def training_step(
    gen: Generator,
    disc: Discriminator,
    gen_optim,
    disc_optim,
    batch: Tuple[Tensor, Tensor, Tensor],
) -> Dict[str, Tensor]:
    line, hint, color = batch
    batch_size = line.shape[0]
    mask = opt.mask_gen(list(hint.shape), X, batch_size // 2)
    hint = hint * mask

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        _color = gen({"line": line, "hint": hint, "training": True})

        logits = disc(color, training=True)
        _logits = disc(_color, training=True)

        adv_loss = losses.binary_crossentropy(_logits, tf.ones_like(_logits))
        l1_loss = losses.l1_loss(_color, color)
        gen_loss = adv_loss + (l1_loss * 100)

        real_loss = losses.binary_crossentropy(logits, tf.ones_like(logits))
        fake_loss = losses.binary_crossentropy(_logits, tf.zeros_like(_logits))
        disc_loss = (real_loss + fake_loss) / 2

    gen_grad = gen_tape.gradient(gen_loss, gen.trainable_variables)
    disc_grad = disc_tape.gradient(disc_loss, disc.trainable_variables)

    gen_optim.apply_gradients(zip(gen_grad, gen.trainable_variables))
    disc_optim.apply_gradients(zip(disc_grad, disc.trainable_variables))

    return {
        # image
        "line": line,
        "hint": hint,
        "color": color,
        "_color": _color,
        # scaler
        "logits": logits,
        "_logits": _logits,
        "adv_loss": adv_loss,
        "l1_loss": l1_loss,
        "gen_loss": gen_loss,
        "real_loss": real_loss,
        "fake_loss": fake_loss,
        "disc_loss": disc_loss,
    }


@tf.function
def test_step(test_batch: batch_type, gen: Generator) -> Dict[str, Tensor]:
    line, hint, color = test_batch
    zero_hint = tf.zeros_like(hint)

    _color_w_hint = gen({"line": line, "hint": hint, "training": False})
    _color_wo_hint = gen({"line": line, "hint": zero_hint, "training": False})

    return {
        # images
        "line": line,
        "hint": hint,
        "color": color,
        "_color_w_hint": _color_w_hint,
        "_color_wo_hint": _color_wo_hint,
    }


def log_step(
    logger, train_images: List[Tensor], test_images: List[Tensor]
) -> None:
    log_image(logger, train_images, Mode.TRAIN)
    test_image = log_image(logger, test_images, Mode.TEST)
    image_save_to_logdir(logger.dir, test_image)


def image_save_to_logdir(logdir: str, image: Union[Tensor, np.ndarray]):
    path = os.path.join(logdir, "image", f"{str(global_step).zfill(8)}.jpg")
    tf.keras.utils.save_img(path, image, "channels_last")


def log_image(
    logger, images: List[Tensor], mode: Mode, max_image_count=8
) -> Tensor:
    images = [
        image[:max_image_count] if len(image) > max_image_count else image
        for image in images
    ]
    images = [
        (tf.concat([image] * 3, -1) if image.shape[-1] == 1 else image)
        for image in images
    ]

    images = tf.concat(images, 1)
    image = tf.concat([image for image in images], 1)
    log_image = wandb.Image(image)
    logger.log({f"images/{mode.value}": log_image})
    return image
