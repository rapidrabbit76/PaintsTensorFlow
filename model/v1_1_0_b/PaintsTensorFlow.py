from tensorflow import keras
import tensorflow as tf
from model.v1_1_0_b.SubNet import GenConv2dBlock, GenUpConv2dBlock, DiscConv2dBlock
import hyperparameter as hp


class Generator_Draft(keras.Model):
    resize = None
    convertUint8 = None

    def __init__(self, resize=None, name=None, convertUint8=None):
        super(Generator_Draft, self).__init__(name=name)
        self.e0 = GenConv2dBlock(hp.gf_dim / 2, 3, 1, name="E0")  # 64
        self.e1 = GenConv2dBlock(hp.gf_dim * 1, 4, 2, name="E1")
        self.e2 = GenConv2dBlock(hp.gf_dim * 1, 3, 1, name="E2")
        self.e3 = GenConv2dBlock(hp.gf_dim * 2, 4, 2, name="E3")
        self.e4 = GenConv2dBlock(hp.gf_dim * 2, 3, 1, name="E4")
        self.e5 = GenConv2dBlock(hp.gf_dim * 4, 4, 2, name="E5")
        self.e6 = GenConv2dBlock(hp.gf_dim * 4, 3, 1, name="E6")
        self.e7 = GenConv2dBlock(hp.gf_dim * 8, 4, 2, name="E7")
        self.e8 = GenConv2dBlock(hp.gf_dim * 8, 3, 1, name="E8")

        self.d8 = GenUpConv2dBlock(hp.gf_dim * 8, 4, 2, name="D8")
        self.d7 = GenConv2dBlock(hp.gf_dim * 4, 3, 1, name="D7")
        self.d6 = GenUpConv2dBlock(hp.gf_dim * 4, 4, 2, name="D6")
        self.d5 = GenConv2dBlock(hp.gf_dim * 2, 3, 1, name="D5")
        self.d4 = GenUpConv2dBlock(hp.gf_dim * 2, 4, 2, name="D4")
        self.d3 = GenConv2dBlock(hp.gf_dim * 1, 3, 1, name="D3")
        self.d2 = GenUpConv2dBlock(hp.gf_dim * 1, 4, 2, name="D2")
        self.d1 = GenConv2dBlock(hp.gf_dim / 2, 3, 1, name="D1")
        self.concat = keras.layers.Concatenate(axis=-1)
        self.last = keras.layers.Conv2D(3, kernel_size=3, strides=1, padding="SAME",
                                        use_bias=False, name="output", activation=tf.nn.tanh,
                                        kernel_initializer=tf.random_normal_initializer(0., 0.02))

        # if resize is not None:
        #     self.resize = keras.layers.Lambda(
        #         lambda image_tensor: tf.image.resize_images(image_tensor, size=(512, 512),
        #                                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))

        self.resize = resize

        if convertUint8 is not None:
            self.convertUint8 = keras.layers.Lambda(lambda tensor: self.cast_uint8(tensor))

    # enable if use Eager Mode
    # @tf.contrib.eager.defun
    def call(self, line, hint, training):
        inputs = tf.concat([line, hint], -1)
        E0 = self.e0(inputs, training=training)
        E1 = self.e1(E0, training=training)
        E2 = self.e2(E1, training=training)
        E3 = self.e3(E2, training=training)
        E4 = self.e4(E3, training=training)
        E5 = self.e5(E4, training=training)

        E6 = self.e6(E5, training=training)
        E7 = self.e7(E6, training=training)
        E8 = self.e8(E7, training=training)

        D8 = self.d8(E7, E8, training=training)
        D7 = self.d7(D8, training=training)
        D6 = self.d6(E6, D7, training=training)
        D5 = self.d5(D6, training=training)
        D4 = self.d4(E4, D5, training=training)
        D3 = self.d3(D4, training=training)
        D2 = self.d2(E2, D3, training=training)
        D1 = self.d1(D2, training=training)

        tensor = self.concat([E0, D1])
        last = self.last(tensor)

        if self.resize is not None:
            last = tf.image.resize_images(last, size=(512, 512),
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return last

    def cast_uint8(self, tensor):
        tensor = (tensor + 1) * 127.5
        return tf.cast(tensor, tf.uint8, name="output")


class Generator(keras.Model):
    convertUint8 = None

    def __init__(self, name=None, convertUint8=None):
        super(Generator, self).__init__(name=name)
        self.e0 = GenConv2dBlock(hp.gf_dim / 2, 3, 1, name="E0")  # 64
        self.e1 = GenConv2dBlock(hp.gf_dim * 1, 4, 2, name="E1")
        self.e2 = GenConv2dBlock(hp.gf_dim * 1, 3, 1, name="E2")
        self.e3 = GenConv2dBlock(hp.gf_dim * 2, 4, 2, name="E3")
        self.e4 = GenConv2dBlock(hp.gf_dim * 2, 3, 1, name="E4")
        self.e5 = GenConv2dBlock(hp.gf_dim * 4, 4, 2, name="E5")
        self.e6 = GenConv2dBlock(hp.gf_dim * 4, 3, 1, name="E6")
        self.e7 = GenConv2dBlock(hp.gf_dim * 8, 4, 2, name="E7")
        self.e8 = GenConv2dBlock(hp.gf_dim * 8, 3, 1, name="E8")

        self.d8 = GenUpConv2dBlock(hp.gf_dim * 8, 4, 2, name="D8")
        self.d7 = GenConv2dBlock(hp.gf_dim * 4, 3, 1, name="D7")
        self.d6 = GenUpConv2dBlock(hp.gf_dim * 4, 4, 2, name="D6")
        self.d5 = GenConv2dBlock(hp.gf_dim * 2, 3, 1, name="D5")
        self.d4 = GenUpConv2dBlock(hp.gf_dim * 2, 4, 2, name="D4")
        self.d3 = GenConv2dBlock(hp.gf_dim * 1, 3, 1, name="D3")
        self.d2 = GenUpConv2dBlock(hp.gf_dim * 1, 4, 2, name="D2")
        self.d1 = GenConv2dBlock(hp.gf_dim / 2, 3, 1, name="D1")
        self.concat = keras.layers.Concatenate(axis=-1)
        self.last = keras.layers.Conv2D(3, kernel_size=3, strides=1, padding="SAME",
                                        use_bias=False, name="output", activation=tf.nn.tanh,
                                        kernel_initializer=tf.random_normal_initializer(0., 0.02))

        if convertUint8 is not None:
            self.convertUint8 = keras.layers.Lambda(lambda tensor: self.cast_uint8(tensor))

    # enable if use Eager Mode
    # @tf.contrib.eager.defun
    def call(self, line, draft, hint, training):
        inputs = tf.concat([line, draft, hint], -1)

        E0 = self.e0(inputs, training=training)
        E1 = self.e1(E0, training=training)
        E2 = self.e2(E1, training=training)
        E3 = self.e3(E2, training=training)
        E4 = self.e4(E3, training=training)
        E5 = self.e5(E4, training=training)

        E6 = self.e6(E5, training=training)
        E7 = self.e7(E6, training=training)
        E8 = self.e8(E7, training=training)

        D8 = self.d8(E7, E8, training=training)
        D7 = self.d7(D8, training=training)
        D6 = self.d6(E6, D7, training=training)
        D5 = self.d5(D6, training=training)
        D4 = self.d4(E4, D5, training=training)
        D3 = self.d3(D4, training=training)
        D2 = self.d2(E2, D3, training=training)
        D1 = self.d1(D2, training=training)

        tensor = self.concat([E0, D1])
        last = self.last(tensor)

        return last

    def cast_uint8(self, tensor):
        tensor = (tensor + 1) * 127.5
        return tf.cast(tensor, tf.uint8, name="output")


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.h0 = DiscConv2dBlock(hp.df_dim / 2, 4, 2)
        self.h1 = DiscConv2dBlock(hp.df_dim / 2, 3, 1)
        self.h2 = DiscConv2dBlock(hp.df_dim * 1, 4, 2)
        self.h3 = DiscConv2dBlock(hp.df_dim * 1, 3, 1)
        self.h4 = DiscConv2dBlock(hp.df_dim * 2, 4, 2)
        self.h5 = DiscConv2dBlock(hp.df_dim * 2, 3, 1)
        self.h6 = DiscConv2dBlock(hp.df_dim * 4, 4, 2)
        self.flatten = keras.layers.Flatten()
        self.last = keras.layers.Dense(2, activation="linear", kernel_initializer=tf.initializers.he_normal())

    # enable if use Eager Mode
    @tf.contrib.eager.defun
    def call(self, inputs, training):
        tensor = self.h0(inputs, training)
        tensor = self.h1(tensor, training)
        tensor = self.h2(tensor, training)
        tensor = self.h3(tensor, training)
        tensor = self.h4(tensor, training)
        tensor = self.h5(tensor, training)
        tensor = self.h6(tensor, training)
        tensor = self.flatten(tensor)  # (?,16384)
        tensor = self.last(tensor)
        return tensor
