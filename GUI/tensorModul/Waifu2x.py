import numpy as np
import tensorflow as tf
import cv2


class Waifu2x:
    __MODEL_PATH__ = "./GUI//src/saved_model/Waifu2x"

    def __init__(self, sess):

        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.2
            self.sess = tf.Session(config=config)
        else:
            self.sess = sess

        self.__build_model()
        self.__zero_init()

    def __build_model(self):
        self.__model = tf.contrib.saved_model.load_keras_model(self.__MODEL_PATH__)

    def __zero_init(self):
        feed = np.zeros(shape=(1, 512, 512, 3))
        self.__model.predict(feed)

    def __image_preprocessing(self, image):
        x, y, _ = image.shape
        image = cv2.resize(image, dsize=(2 * y, 2 * x))
        return np.array([image]) / 255.0

    def __image_postprocessing(self, image):
        image = np.clip(image[0], 0, 1) * 255
        return image.astype(np.uint8)

    def upscale(self, image):
        image = self.__image_preprocessing(image=image)
        image = self.__model.predict(image)
        image = self.__model.predict(image)
        return self.__image_postprocessing(image=image)
