import argparse

import wandb
from trainer import draft_train


def main():
    parser = argparse.ArgumentParser()
    # project
    parser.add_argument("--project_name", type=str, default="PaintsTensorflow")
    parser.add_argument("--logdir", type=str, default="experiment")
    parser.add_argument("--mode", type=str, choices=["draft", "colorization"])
    # data
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)

    # model
    parser.add_argument("--g_dim", type=int, default=64)
    parser.add_argument("--d_dim", type=int, default=64)

    # training
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--beta_1", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=1)
    parser.add_argument("--sigma", type=float, default=0.005)

    args = parser.parse_args()

    ########################## logger ##########################
    logger = wandb.init(
        config=args,
        name=args.mode,
        sync_tensorboard=True,
        save_code=True,
    )
    trainer = draft_train
    trainer(args, logger)


if __name__ == "__main__":
    main()