import tensorflow as tf
import numpy as np
import cv2

from dataset.Datasets import Datasets_512
import utils

__DRAFT_MODEL_PATH__ = "./saved_model/PaintsTensorFlowDraftModel"
__MODEL_PATH__ = "./saved_model/PaintsTensorFlowModel"


class PaintsTensorflowTest:

    def __init__(self, log=False):
        self.__pre_generator = tf.contrib.saved_model.load_keras_model(__DRAFT_MODEL_PATH__)
        self.__generator = tf.contrib.saved_model.load_keras_model(__MODEL_PATH__)
        if log: tf.summary.FileWriter("./log", tf.Session().graph)

    def test(self, itr=100, zero_hint=False):
        sess = tf.Session()
        dataSets = Datasets_512(batch_size=1)
        train_sets, test_sets = dataSets.buildDataSets()

        train_sets = train_sets.make_initializable_iterator()
        sess.run(train_sets.initializer)
        train_next = train_sets.get_next()

        for _ in range(itr):
            line_128, hint_128, image, line, _ = sess.run(train_next)

            hint = np.ones_like(hint_128)
            hint += 1

            if zero_hint:
                draft = self.__pre_generator.predict([line_128, hint])
            else:
                draft = self.__pre_generator.predict([line_128, hint_128])

            draft = tf.image.resize_images(draft, size=(512, 512),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            draft = sess.run(draft)
            outputs = self.__generator.predict([line, draft])
            outputs = np.concatenate([outputs, image], 2)
            outputs = utils.convert2uint8(outputs)[0]
            cv2.imshow("", cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)


if __name__ == '__main__':
    test = PaintsTensorflowTest()
    test.test(3)
