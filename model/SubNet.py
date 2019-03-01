import tensorflow as tf

__INITIALIZER__ = tf.random_normal_initializer(0., 0.02)
__MOMENTUM__ = 0.9
__EPSILON__ = 1e-5


def res_net_block_v2(inputs, filters):
    with tf.name_scope("ResNetBlock"):
        shortcut = inputs
        tensor = tf.keras.layers.BatchNormalization()(inputs)
        tensor = tf.keras.layers.ReLU()(tensor)
        tensor = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="SAME")(tensor)

        tensor = tf.keras.layers.BatchNormalization()(tensor)
        tensor = tf.keras.layers.ReLU()(tensor)
        tensor = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="SAME")(tensor)
        tensor = tf.keras.layers.add([shortcut, tensor])
    return tensor


def GenConvBlock(inputs, filters, k, s, res_net_block=True, name="GenConvBlock"):
    filters = int(filters)
    with tf.name_scope(name):
        tensor = tf.keras.layers.Conv2D(filters=filters, kernel_size=k, strides=s, use_bias=False,
                                        padding="SAME", kernel_initializer=__INITIALIZER__)(inputs)

        if res_net_block:
            tensor = res_net_block_v2(tensor, filters)
        else:
            tensor = tf.keras.layers.BatchNormalization(momentum=__MOMENTUM__, epsilon=__EPSILON__)(tensor)
            tensor = tf.keras.layers.LeakyReLU()(tensor)

        return tensor


def GenUpConvBlock(inputs_a, inputs_b, filters, k, s, res_net_block=True, name="GenUpConvBlock"):
    filters = int(filters)
    with tf.name_scope(name):
        tensor = tf.keras.layers.Concatenate(3)([inputs_a, inputs_b])
        tensor = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=k, strides=s, use_bias=False,
                                                 padding="SAME", kernel_initializer=__INITIALIZER__)(tensor)

        if res_net_block:
            tensor = res_net_block_v2(tensor, filters)
        else:
            tensor = tf.keras.layers.BatchNormalization(momentum=__MOMENTUM__, epsilon=__EPSILON__)(tensor)
            tensor = tf.keras.layers.ReLU()(tensor)

        return tensor


class DisConvBlock(tf.keras.Model):
    def __init__(self, filters, k, s, apply_bat_norm=True, name=None):
        super(DisConvBlock, self).__init__(name=name)
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)
        self.apply_bat_norm = apply_bat_norm
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=k, strides=s,
                                           padding="SAME", kernel_initializer=initializer)
        if self.apply_bat_norm:
            self.bn = tf.keras.layers.BatchNormalization(momentum=__MOMENTUM__, epsilon=__EPSILON__)

        self.act = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, inputs, training):
        tensor = self.conv(inputs)

        if self.apply_bat_norm:
            tensor = self.bn(tensor, training=training)

        tensor = self.act(tensor)
        return tensor


def tf_int_round(num):
    return tf.cast(tf.round(num), dtype=tf.int32)


class resize_layer(tf.keras.layers.Layer):
    def __init__(self, size=(512, 512), **kwargs, ):
        super(resize_layer, self).__init__(**kwargs)
        (self.height, self.width) = size

    def build(self, input_shape):
        super(resize_layer, self).build(input_shape)

    def call(self, x, method="nearest"):
        height = 512
        width = 512

        if method == "nearest":
            return tf.image.resize_nearest_neighbor(x, size=(height, width))
        elif method == "bicubic":
            return tf.image.resize_bicubic(x, size=(height, width))
        elif method == "bilinear":
            return tf.image.resize_bilinear(x, size=(height, width))

    def get_output_shape_for(self, input_shape):
        return (self.input_shape[0], 512, 512, 3)
