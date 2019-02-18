import tensorflow as tf
from model.SubNet import GenConvBlock, GenUpConvBlock, DisConvBlock
import hyperparameter as hp


def Generator(inputs_size=None, res_net_block=True, name="PaintsTensorFlow"):
    inputs_line = tf.keras.Input(shape=[inputs_size, inputs_size, 1], dtype=tf.float32, name="inputs_line")
    inputs_hint = tf.keras.Input(shape=[inputs_size, inputs_size, 3], dtype=tf.float32, name="inputs_hint")
    tensor = tf.keras.layers.Concatenate(3)([inputs_line, inputs_hint])

    e0 = GenConvBlock(tensor, hp.gf_dim / 2, 3, 1, res_net_block=res_net_block, name="E0")  # 64
    e1 = GenConvBlock(e0, hp.gf_dim * 1, 4, 2, res_net_block=res_net_block, name="E1")
    e2 = GenConvBlock(e1, hp.gf_dim * 1, 3, 1, res_net_block=res_net_block, name="E2")
    e3 = GenConvBlock(e2, hp.gf_dim * 2, 4, 2, res_net_block=res_net_block, name="E3")
    e4 = GenConvBlock(e3, hp.gf_dim * 2, 3, 1, res_net_block=res_net_block, name="E4")
    e5 = GenConvBlock(e4, hp.gf_dim * 4, 4, 2, res_net_block=res_net_block, name="E5")
    e6 = GenConvBlock(e5, hp.gf_dim * 4, 3, 1, res_net_block=res_net_block, name="E6")
    e7 = GenConvBlock(e6, hp.gf_dim * 8, 4, 2, res_net_block=res_net_block, name="E7")
    e8 = GenConvBlock(e7, hp.gf_dim * 8, 3, 1, res_net_block=res_net_block, name="E8")

    d8 = GenUpConvBlock(e7, e8, hp.gf_dim * 8, 4, 2, res_net_block=res_net_block, name="D8")
    d7 = GenConvBlock(d8, hp.gf_dim * 4, 3, 1, res_net_block=res_net_block, name="D7")
    d6 = GenUpConvBlock(e6, d7, hp.gf_dim * 4, 4, 2, res_net_block=res_net_block, name="D6")
    d5 = GenConvBlock(d6, hp.gf_dim * 2, 3, 1, res_net_block=res_net_block, name="D5")
    d4 = GenUpConvBlock(e4, d5, hp.gf_dim * 2, 4, 2, res_net_block=res_net_block, name="D4")
    d3 = GenConvBlock(d4, hp.gf_dim * 1, 3, 1, res_net_block=res_net_block, name="D3")
    d2 = GenUpConvBlock(e2, d3, hp.gf_dim * 1, 4, 2, res_net_block=res_net_block, name="D2")
    d1 = GenConvBlock(d2, hp.gf_dim / 2, 3, 1, res_net_block=res_net_block, name="D1")

    tensor = tf.keras.layers.Concatenate(3)([e0, d1])
    outputs = tf.keras.layers.Conv2D(hp.c_dim, kernel_size=3, strides=1, padding="SAME",
                                     use_bias=True, name="output", activation=tf.nn.tanh,
                                     kernel_initializer=tf.random_normal_initializer(0., 0.02))(tensor)

    inputs = [inputs_line, inputs_hint]
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.h0 = DisConvBlock(hp.df_dim / 2, 4, 2)
        self.h1 = DisConvBlock(hp.df_dim / 2, 3, 1)
        self.h2 = DisConvBlock(hp.df_dim * 1, 4, 2)
        self.h3 = DisConvBlock(hp.df_dim * 1, 3, 1)
        self.h4 = DisConvBlock(hp.df_dim * 2, 4, 2)
        self.h5 = DisConvBlock(hp.df_dim * 2, 3, 1)
        self.h6 = DisConvBlock(hp.df_dim * 4, 4, 2)
        self.flatten = tf.keras.layers.Flatten()
        self.last = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=tf.initializers.he_normal())

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
