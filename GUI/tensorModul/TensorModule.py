import tensorflow as tf

from GUI.tensorModul.SketchKeras import SketchKeras
from GUI.tensorModul.PaintsTensorFlow import PaintsTensorFlow
from GUI.tensorModul.Waifu2x import Waifu2x


class TensorModule:

    def __init__(self, sess=None):
        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
        else:
            self.sess = sess

        self.__paints_tf = PaintsTensorFlow(sess=self.sess)
        self.__sketch_keras = SketchKeras(sess=self.sess)
        self.__waifu2x = Waifu2x(sess=self.sess)

    def pred_image(self, line, hint):
        return self.__paints_tf.pred_image(line=line, hint=hint)

    def get_line(self, image):
        return self.__sketch_keras.get_line(image=image)

    def upscale(self, image):
        return self.__waifu2x.upscale(image=image)