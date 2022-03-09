import os
from typing import Dict, List, Tuple, Union

import numpy as np
import opt
import scipy.stats as stats
import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import wandb
from dataset import draft_dataset
from models import Discriminator, Generator
from tensorflow import Tensor
from tqdm import tqdm

global global_step
global_step = 0


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

    train_dataset = draft_dataset.build_dataset(
        root_dir=os.path.join(args.root_dir, "train"),
        image_size=args.image_size,
        batch_size=args.batch_size,
        train=True,
    )
    test_dataset = draft_dataset.build_dataset(
        root_dir=os.path.join(args.root_dir, "test"),
        image_size=args.image_size,
        batch_size=8,
        train=False,
    )
    test_batch = next(iter(test_dataset))
    line, hint, color = test_batch
    mask = opt.mask_gen(list(hint.shape), X)
    hint = hint * mask
    test_batch = (line, hint, color)

    ########################## model ##########################

    gen = Generator(args.g_dim)
    disc = Discriminator(args.d_dim)

    ########################## optim ##########################
    gen_optim = optimizers.Adam(args.lr, beta_1=args.beta_1)
    disc_optim = optimizers.Adam(args.lr, beta_1=args.beta_1)

    ######################## training ##########################
    for epoch in range(args.epochs):
        training_loop(
            train_dataset, test_batch, gen, disc, gen_optim, disc_optim
        )

    #################### artifacts loging ######################
    artifacts_path = os.path.join(logger.dir, f"{args.mode}.h5")
    gen.save_weights(artifacts_path)
    gen_artifacts = wandb.Artifact(args.mode, type="model")
    gen_artifacts.add_file(artifacts_path, "weight")
    logger.log_artifact(gen_artifacts)


def training_loop(train_dataset, test_batch, gen, disc, gen_optim, disc_optim):
    global global_step
    dataset = tqdm(train_dataset)
    for batch_idx, batch in enumerate(dataset):
        global_step += 1
        training_info = training_step(gen, disc, gen_optim, disc_optim, batch)

        if batch_idx % 10 == 0:
            log_scalers(training_info)

        if batch_idx % 100 == 0:
            images = test_step(test_batch, gen)
            log_images(
                [
                    training_info["line"],
                    training_info["hint"],
                    training_info["_color"],
                    training_info["color"],
                ],
                "train",
            )
            test_images = log_images(images, "test")
            image_save(test_images)

    test_images = log_images(images, "test")
    image_save(test_images)


def image_save(image: Union[Tensor, np.ndarray]):
    path = os.path.join(
        logger.dir,
        "image",
        f"{str(global_step).zfill(8)}.jpg",
    )
    tf.keras.utils.save_img(path, image, "channels_last")
    return 0


def log_scalers(training_info):
    log_dict = {
        "real_prob": opt.logits_to_prob(training_info["logits"]),
        "fake_prob": opt.logits_to_prob(training_info["_logits"]),
        "adv_loss": training_info["adv_loss"],
        "l1_loss": training_info["l1_loss"],
        "gen_loss": training_info["gen_loss"],
        "disc_real_loss": training_info["disc_real_loss"],
        "disc_fake_loss": training_info["disc_fake_loss"],
        "disc_loss": training_info["disc_loss"],
    }
    logger.log(log_dict)
    return log_dict


def log_images(images: List[Tensor], name: str) -> Tensor:
    images = tf.concat(
        [
            (tf.concat([image] * 3, -1) if image.shape[-1] == 1 else image)
            for image in images
        ],
        1,
    )
    image = tf.concat([image for image in images], 1)
    log_image = wandb.Image(image)
    logger.log({f"images/{name}": log_image})
    return image


@tf.function
def test_step(
    test_batch: Tuple[Tensor, Tensor, Tensor], gen: Generator
) -> List[Tensor]:
    line, hint, color = test_batch
    zero_hint = tf.zeros_like(hint) - 1
    training = False

    _color_w_hint = gen(line, hint, training=training)
    _color_wo_hint = gen(line, zero_hint, training=training)

    # plotting
    return [line, hint, _color_w_hint, _color_wo_hint, color]


@tf.function
def training_step(
    gen: Generator,
    disc: Discriminator,
    gen_optim,
    disc_optim,
    batch: Tuple[Tensor, Tensor, Tensor],
) -> Dict[str, Tensor]:
    training = True
    line, hint, color = batch
    mask = opt.mask_gen(list(hint.shape), X)
    hint = hint * mask

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        _color = gen(line, hint, training=training)

        logits = disc(color, training=training)
        _logits = disc(_color, training=training)

        adv_loss, l1_loss = generator_loss(_logits, _color, color)
        gen_loss = adv_loss + l1_loss * 100

        disc_real_loss, disc_fake_loss = discriminator_loss(logits, _logits)
        disc_loss = (disc_real_loss + disc_fake_loss) / 2

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
        "disc_real_loss": disc_real_loss,
        "disc_fake_loss": disc_fake_loss,
        "disc_loss": disc_loss,
    }


def generator_loss(
    _logits: Tensor, _color: Tensor, color: Tensor
) -> Tuple[Tensor, Tensor]:
    adv_loss = tf.reduce_mean(
        tf.keras.metrics.binary_crossentropy(
            tf.ones_like(_logits),
            _logits,
            from_logits=True,
        )
    )
    l1_loss = tf.reduce_mean(tf.abs(_color - color))
    return adv_loss, l1_loss


def discriminator_loss(
    logits: Tensor, _logits: Tensor
) -> Tuple[Tensor, Tensor]:
    disc_real_loss = tf.reduce_mean(
        tf.keras.metrics.binary_crossentropy(
            tf.ones_like(logits),
            logits,
            from_logits=True,
        )
    )
    disc_fake_loss = tf.reduce_mean(
        tf.keras.metrics.binary_crossentropy(
            tf.zeros_like(_logits),
            _logits,
            from_logits=True,
        )
    )
    return disc_real_loss, disc_fake_loss
