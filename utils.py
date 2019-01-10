import tensorflow as tf
import numpy as np
import copy

import cv2
import os
from glob import glob
from random import randint
import hyperparameter as hp


def get_line(imgs):
    def img_liner(img):
        k = 3
        kernal = np.ones((k, k), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dilated = cv2.dilate(gray, kernal, iterations=1)
        diff = cv2.absdiff(dilated, gray)
        img = 255 - diff
        return img

    lines = np.array([img_liner(l) for l in imgs])
    return np.expand_dims(lines, 3)


def convert2f32(img):
    img = img.astype(np.float32)
    return (img / 127.5) - 1.0


def convert2uint8(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8)


def convertRGB(imgs):
    imgs = np.asarray(imgs, np.uint8)
    return np.array([cv2.cvtColor(img, cv2.COLOR_YUV2RGB) for img in imgs])


def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
