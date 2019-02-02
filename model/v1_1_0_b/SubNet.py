import tensorflow as tf
from tensorflow import keras

__MOMENTUM__ = 0.9
__EPSILON__ = 1e-5


def batch_norm():
    return keras.layers.BatchNormalization(momentum=__MOMENTUM__, epsilon=__EPSILON__)


class GenConv2dBlock(tf.keras.Model):

    def __init__(self, filters, k, s, name=None):
        super(GenConv2dBlock, self).__init__(name=name)
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)

        self.conv = keras.layers.Conv2D(filters=filters, kernel_size=k, strides=s, use_bias=False,
                                        padding="SAME", kernel_initializer=initializer)
        self.resblock = ResNetBlock(filters=filters)
        self.act = keras.layers.LeakyReLU(alpha=0.2)

    def call(self, inputs, training):
        tensor = self.conv(inputs)
        tensor = self.resblock(tensor, training=training)
        tensor = self.act(tensor)
        return tensor


class GenUpConv2dBlock(tf.keras.Model):

    def __init__(self, filters, k, s, name=None):
        super(GenUpConv2dBlock, self).__init__(name=name)

        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)

        self.concat = keras.layers.Concatenate(axis=-1)
        self.up_conv = keras.layers.Conv2DTranspose(filters=filters, kernel_size=k, strides=s, use_bias=False,
                                                    padding="SAME", kernel_initializer=initializer)
        self.resblock = ResNetBlock(filters=filters)
        self.act = keras.layers.ReLU()

    def call(self, x1, x2, training):
        inputs = self.concat([x1, x2])
        tensor = self.up_conv(inputs)
        tensor = self.resblock(tensor, training=training)
        tensor = self.act(tensor)
        return tensor


class ResNetBlock(tf.keras.Model):
    def __init__(self, filters):
        super(ResNetBlock, self).__init__(name="ResNetBlock")

        self.conv2d_1 = keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="SAME")
        self.bn_1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()

        self.conv2d_2 = keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="SAME")
        self.bn_2 = keras.layers.BatchNormalization()
        self.relu_2 = keras.layers.ReLU()

    def call(self, inputs, training):
        shortcut = inputs

        tensor = self.conv2d_1(inputs)
        tensor = self.bn_1(tensor, training=training)
        tensor = self.relu(tensor)

        tensor = self.conv2d_2(tensor)
        tensor = self.bn_2(tensor, training=training)

        tensor = shortcut + tensor
        tensor = self.relu_2(tensor)
        return tensor


class DiscConv2dBlock(tf.keras.Model):
    def __init__(self, filters, k, s):
        super(DiscConv2dBlock, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)
        self.conv = keras.layers.Conv2D(filters=filters, kernel_size=k, strides=s,
                                        padding="SAME", kernel_initializer=initializer)
        self.bn = batch_norm()
        self.act = keras.layers.LeakyReLU(alpha=0.2)

    def call(self, inputs, training):
        tensor = self.conv(inputs)
        tensor = self.bn(tensor, training=training)
        tensor = self.act(tensor)
        return tensor
