from model.SubNet import *
from tensorflow import keras
import hyperparameter as hp


class Generator(keras.Model):
    def __init__(self, resize=None, name=None, convertUint8=None):
        super(Generator, self).__init__(name=name)
        self.e0 = genConv2dLayer(hp.gf_dim / 2, 3, 1, name="E0")  # 64
        self.e1 = genConv2dLayer(hp.gf_dim * 1, 4, 2, name="E1")
        self.e2 = genConv2dLayer(hp.gf_dim * 1, 3, 1, name="E2")
        self.e3 = genConv2dLayer(hp.gf_dim * 2, 4, 2, name="E3")
        self.e4 = genConv2dLayer(hp.gf_dim * 2, 3, 1, name="E4")
        self.e5 = genConv2dLayer(hp.gf_dim * 4, 4, 2, name="E5")
        self.e6 = genConv2dLayer(hp.gf_dim * 4, 3, 1, name="E6")
        self.e7 = genConv2dLayer(hp.gf_dim * 8, 4, 2, name="E7")
        self.e8 = genConv2dLayer(hp.gf_dim * 8, 3, 1, name="E8")

        self.d8 = genDeConv2dLayer(hp.gf_dim * 8, 4, 2, name="D8")
        self.d7 = genConv2dLayer(hp.gf_dim * 4, 3, 1, name="D7")
        self.d6 = genDeConv2dLayer(hp.gf_dim * 4, 4, 2, name="D6")
        self.d5 = genConv2dLayer(hp.gf_dim * 2, 3, 1, name="D5")
        self.d4 = genDeConv2dLayer(hp.gf_dim * 2, 4, 2, name="D4")
        self.d3 = genConv2dLayer(hp.gf_dim * 1, 3, 1, name="D3")
        self.d2 = genDeConv2dLayer(hp.gf_dim * 1, 4, 2, name="D2")
        self.d1 = genConv2dLayer(hp.gf_dim / 2, 3, 1, name="D1")

        self.last = keras.layers.Conv2D(3, kernel_size=3, strides=1, padding="SAME",
                                        use_bias=True, name="output", activation=tf.nn.tanh,
                                        kernel_initializer=tf.random_normal_initializer(0., 0.02))
        self.resize = resize
        self.convertUint8 = convertUint8

    # if use Eager Mode
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
        tensor = tf.concat([E0, D1], -1)
        last = self.last(tensor)

        if self.resize is not None:
            last = tf.image.resize_images(last, (512, 512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if self.convertUint8 is not None:
            last = (last + 1) * 127.5
            last = tf.cast(last,tf.uint8,name="output")

        return last


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.h0 = DiscConv2d(hp.df_dim / 2, 4, 2)
        self.h1 = DiscConv2d(hp.df_dim / 2, 3, 1)
        self.h2 = DiscConv2d(hp.df_dim * 1, 4, 2)
        self.h3 = DiscConv2d(hp.df_dim * 1, 3, 1)
        self.h4 = DiscConv2d(hp.df_dim * 2, 4, 2)
        self.h5 = DiscConv2d(hp.df_dim * 2, 3, 1)
        self.h6 = DiscConv2d(hp.df_dim * 4, 4, 2)
        self.pad = L.ZeroPadding2D()
        self.last = L.Conv2D(1, 4, 1, kernel_initializer=tf.random_normal_initializer(0., 0.02))

    # if use Eager Mode
    # @tf.contrib.eager.defun
    def call(self, inputs, training):
        tensor = self.h0(inputs, training)
        tensor = self.h1(tensor, training)
        tensor = self.h2(tensor, training)
        tensor = self.h3(tensor, training)
        tensor = self.h4(tensor, training)
        tensor = self.h5(tensor, training)
        tensor = self.h6(tensor, training)
        tensor = self.pad(tensor)
        tensor = self.last(tensor)
        return tensor
