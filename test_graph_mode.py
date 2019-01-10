import numpy as np
import cv2
from dataset.Datasets import Datasets_512
from model.PaintsTensorFlow import *


class PaintsTensorflowTest:
    def __init__(self):
        self.__pre_generator = Generator(resize=True, name="predPaintsTensorFlow")
        self.__generator = Generator(name="PaintsTensorFlow", convertUint8=True)

    def test(self):
        pred_model_path = "./ckpt/PaintsTensorFlow/PredPaintsTensorFlow.h5"
        model_path = "./ckpt/PaintsTensorFlow/PaintsTensorFlow.h5"

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        dataSets = Datasets_512(batch_size=hp.batch_size)
        train_sets, test_sets = dataSets.buildDataSets()

        train_sets = train_sets.make_initializable_iterator()
        sess.run(train_sets.initializer)
        train_sets = train_sets.get_next()

        line = keras.Input(shape=(128, 128, 1), dtype=tf.float32, name="input_line_128")
        hint = keras.Input(shape=(128, 128, 3), dtype=tf.float32, name="input_hint_128")
        outputs = self.__pre_generator(line, hint, False)
        pred_model = keras.Model(inputs=[line, hint], outputs=outputs)

        line = keras.Input(shape=(512, 512, 1), dtype=tf.float32, name="input_line_512")
        hint = keras.Input(shape=(512, 512, 3), dtype=tf.float32, name="input_hint_512")
        outputs = self.__generator(line, hint, False)
        model = keras.Model(inputs=[line, hint], outputs=outputs)

        sess.run(tf.global_variables_initializer())

        self.__pre_generator.load_weights(pred_model_path)
        self.__generator.load_weights(model_path)
        tf.summary.FileWriter("./log", sess.graph)

        for _ in range(2):
            line_128, hint_128, image, line, hint = sess.run(train_sets)
            hint = np.ones_like(hint_128)
            hint += 1
            hint = pred_model.predict([line_128, hint])
            images = model.predict([line, hint])

            for img in images:
                cv2.imshow("", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)


if __name__ == '__main__':
    test = PaintsTensorflowTest()
    test.test()
