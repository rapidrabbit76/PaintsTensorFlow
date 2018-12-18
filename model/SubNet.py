import tensorflow as tf
import tensorflow.keras.layers as L


class genConv2dLayer(tf.keras.Model):

    def __init__(self, filters, k, s, applyBatNorm=True, name=None):
        super(genConv2dLayer, self).__init__(name=name)
        self.applyBn = applyBatNorm
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)
        self.conv = L.Conv2D(filters=filters, kernel_size=k, strides=s, padding="SAME",
                             kernel_initializer=initializer)
        if self.applyBn:
            self.bn = L.BatchNormalization(momentum=0.9,
                                           epsilon=1e-5,
                                           scale=True)

    def call(self, inputs, training):
        tensor = self.conv(inputs)
        if self.applyBn:
            tensor = self.bn(tensor, training=training)
        tensor = tf.nn.leaky_relu(tensor)
        return tensor


class genDeConv2dLayer(tf.keras.Model):

    def __init__(self, filters, k, s, name=None):
        super(genDeConv2dLayer, self).__init__(name=name)

        initializer = tf.random_normal_initializer(0., 0.02)
        self.concat = tf.concat
        filters = int(filters)
        self.up_conv = L.Conv2DTranspose(filters=filters, kernel_size=k, strides=s, padding="SAME",
                                         kernel_initializer=initializer)
        self.bn = L.BatchNormalization(momentum=0.9,
                                       epsilon=1e-5,
                                       scale=True)

    def call(self, x1, x2, training):
        inputs = self.concat([x1, x2], -1)
        tensor = self.up_conv(inputs)
        tensor = self.bn(tensor, training=training)
        tensor = tf.nn.relu(tensor)
        return tensor


class DiscConv2d(tf.keras.Model):
    def __init__(self, filters, k, s, applyBatNorm=True):
        super(DiscConv2d, self).__init__()
        self.applyBn = applyBatNorm
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)
        self.conv = L.Conv2D(filters=filters, kernel_size=k, strides=s, padding="SAME",
                             kernel_initializer=initializer)
        if self.applyBn:
            self.bn = L.BatchNormalization(momentum=0.9,
                                           epsilon=1e-5,
                                           scale=True)

    def call(self, inputs, training):
        tensor = self.conv(inputs)
        if self.applyBn:
            tensor = self.bn(tensor, training=training)
        tensor = tf.nn.leaky_relu(tensor)
        return tensor
