import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

import numpy as np
import cv2
from dataset.Datasets import Datasets, Datasets_512
from model.PaintsTensorFlow import *


class PaintsTensorflowTest:

    def __init__(self):
        def model_init(model, size):
            zero_0 = np.zeros(shape=[1, size, size, 1], dtype=np.float32)
            zero_1 = np.zeros(shape=[1, size, size, 3], dtype=np.float32)
            model(zero_0, zero_1, False)

        pred_model_path = "./ckpt/PaintsTensorFlow/PredPaintsTensorFlow.h5"
        model_path = "./ckpt/PaintsTensorFlow/PaintsTensorFlow.h5"

        self.__pre_generator = Generator(resize=True, name="predPaintsTensorFlow")
        self.__generator = Generator(name="PaintsTensorFlow", convertUint8=True)

        model_init(self.__pre_generator, 128)
        model_init(self.__generator, 512)

        self.__pre_generator.load_weights(pred_model_path)
        self.__generator.load_weights(model_path)

    def test(self):
        dataSets = Datasets_512(batch_size=hp.batch_size)
        train_sets, test_sets = dataSets.buildDataSets()

        for line_128, hint_128, image, line, hint in test_sets.take(2):
            hint = hint.numpy()
            hint = np.ones_like(hint_128)
            hint += 1
            pred = self.__pre_generator(line_128, hint, False)
            outputs = self.__generator(line, pred, training=False)

            for img in outputs.numpy():
                cv2.imshow("", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)

        self.__pre_generator.summary()
        self.__generator.summary()


if __name__ == '__main__':
    test = PaintsTensorflowTest()
    test.test()
