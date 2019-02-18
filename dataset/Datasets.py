import cv2
import numpy as np
import tensorflow as tf
import hyperparameter as hp
from glob import glob


class Datasets:
    def __init__(self, prefetch=-1, batch_size=1, shuffle=False):
        self.prefetch = prefetch
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _preprocess(self, image, line, training):
        if training:
            if np.random.rand() < 0.5:
                image = cv2.flip(image, 0)
                line = cv2.flip(line, 0)
                line = np.expand_dims(line, 3)

            if np.random.rand() < 0.5:
                image = cv2.flip(image, 1)
                line = cv2.flip(line, 1)
                line = np.expand_dims(line, 3)

        return image, line, self._buildHint_resize(image)

    def _buildHint_resize(self, image):
        random = np.random.rand
        hint = np.ones_like(image)
        hint += 1
        leak_count = np.random.randint(16, 120)

        if random() < 0.4:
            leak_count = 0
        elif random() < 0.7:
            leak_count = np.random.randint(2, 16)

        # leak position
        x = np.random.randint(1, image.shape[0] - 1, leak_count)
        y = np.random.randint(1, image.shape[1] - 1, leak_count)

        def paintCel(i):
            color = image[x[i]][y[i]]
            hint[x[i]][y[i]] = color

            if random() > 0.5:
                hint[x[i]][y[i] + 1] = color
                hint[x[i]][y[i] - 1] = color

            if random() > 0.5:
                hint[x[i] + 1][y[i]] = color
                hint[x[i] - 1][y[i]] = color

        for i in range(leak_count):
            paintCel(i)

        return hint

    def convert2float(self, image):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        return image

    def __line_threshold(self, line):
        if np.random.rand() < 0.3:
            line = np.reshape(line, newshape=(512, 512))
            _, line = cv2.threshold(line, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            line = np.reshape(line, newshape=(512, 512, 1))
        return line

    def loadImage(self, imagePath, linePath, train):
        image = tf.read_file(imagePath)
        image = tf.image.decode_jpeg(image, channels=3)

        line = tf.read_file(linePath)
        line = tf.image.decode_jpeg(line, channels=1)

        image = tf.image.resize_images(image, (128, 128), method=3)
        line = tf.image.resize_images(line, (128, 128), method=3)

        image = self.convert2float(image)
        line = self.convert2float(line)

        image, line, hint = tf.py_func(self._preprocess,
                                       [image, line, train],
                                       [tf.float32, tf.float32, tf.float32])

        return image, line, hint

    def buildDataSets(self):
        def build_dataSets(image, line, shuffle=False, isTrain=False):
            image = glob(image)
            image.sort()
            line = glob(line)
            line.sort()

            if shuffle is False and isTrain is False:
                image.reverse()
                line.reverse()

            hp.batch_steps = int(len(line) / self.batch_size)
            datasets = tf.data.Dataset.from_tensor_slices((image, line))
            datasets = datasets.map(lambda x, y: self.loadImage(x, y, isTrain))
            datasets = datasets.batch(self.batch_size)

            if shuffle:
                datasets = datasets.shuffle(100)

            return datasets

        testDatasets = build_dataSets(hp.test_image_datasets_path,
                                      hp.test_line_datasets_path,
                                      shuffle=False, isTrain=False)

        trainDatasets = build_dataSets(hp.train_image_datasets_path,
                                       hp.train_line_datasets_path,
                                       shuffle=self.shuffle, isTrain=True)

        return trainDatasets, testDatasets


class Datasets_512(Datasets):
    def _flip(self, image, line, training):
        if training:
            if np.random.rand() < 0.5:
                image = cv2.flip(image, 0)
                line = cv2.flip(line, 0)
                line = np.expand_dims(line, 3)

            if np.random.rand() < 0.5:
                image = cv2.flip(image, 1)
                line = cv2.flip(line, 1)
                line = np.expand_dims(line, 3)

        return image, line

    def _buildHint(self, image):
        random = np.random.rand
        hint = np.ones_like(image)
        hint += 1
        leak_count = np.random.randint(16, 128)

        # leak position
        x = np.random.randint(1, image.shape[0] - 1, leak_count)
        y = np.random.randint(1, image.shape[1] - 1, leak_count)

        def paintCel(i):
            color = image[x[i]][y[i]]
            hint[x[i]][y[i]] = color

            if random() > 0.5:
                hint[x[i]][y[i] + 1] = color
                hint[x[i]][y[i] - 1] = color

            if random() > 0.5:
                hint[x[i] + 1][y[i]] = color
                hint[x[i] - 1][y[i]] = color

        for i in range(leak_count):
            paintCel(i)
        return hint

    def loadImage(self, imagePath, linePath, train):
        image = tf.read_file(imagePath)
        image = tf.image.decode_jpeg(image, channels=3)
        line = tf.read_file(linePath)
        line = tf.image.decode_jpeg(line, channels=1)

        image_128 = tf.image.resize_images(image, (128, 128), method=3)
        line_128 = tf.image.resize_images(line, (128, 128), method=3)

        image = self.convert2float(image)
        line = self.convert2float(line)
        image_128 = self.convert2float(image_128)
        line_128 = self.convert2float(line_128)

        hint_128 = tf.py_func(self._buildHint,
                              [image_128],
                              tf.float32)
        hint_128.set_shape(shape=image_128.shape)
        hint = tf.image.resize_images(hint_128, (512, 512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return line_128, hint_128, image, line, hint
