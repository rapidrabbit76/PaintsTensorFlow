from trainer.draft import draft_train
from trainer.colorization import colorization_train

TRAINING_TABLE = {
    "draft": draft_train,
    "colorization": colorization_train,
}
