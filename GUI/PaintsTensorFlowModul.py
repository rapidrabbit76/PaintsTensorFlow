import tensorflow as tf
from GUI.QtUtil import *
from model.PaintsTensorFlow import Generator
from tensorflow import keras
from GUI.SketchKeras import SketchKeras


class PaintsTensorFlowModul:
    pred_model_path = "./GUI/model/PaintsTensorFlow/PredPaintsTensorFlow.h5"
    model_path = "./GUI/model/PaintsTensorFlow/PaintsTensorFlow.h5"

    def __init__(self, sess=None):
        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.2
            self.sess = tf.Session(config=config)
        else:
            self.sess = sess

        self.__build_model()

    def __build_model(self):
        ''' Load & build tensorflow graph
        tensors Name:

            Draft_net:
                Tensor("input_line_128:0", shape=(?, 128, 128, 1), dtype=float32)
                Tensor("input_hint_128:0", shape=(?, 128, 128, 3), dtype=float32)
                Tensor("predPaintTensorFlow/resize_images/ResizeNearestNeighbor:0", shape=(?, 512, 512, 3), dtype=float32)

            Generator:
                Tensor("input_line_512:0", shape=(?, 512, 512, 1), dtype=float32)
                Tensor("input_hint_512:0", shape=(?, 512, 512, 3), dtype=float32)
                Tensor("PaintTensorFlow/output_1:0", shape=(?, 512, 512, 3), dtype=uint8)
        '''

        pre_generator = Generator(resize=True, name="predPaintsTensorFlow")
        generator = Generator(name="PaintsTensorFlow", convertUint8=True)
        line = keras.Input(shape=(128, 128, 1), dtype=tf.float32, name="input_line_128")
        hint = keras.Input(shape=(128, 128, 3), dtype=tf.float32, name="input_hint_128")
        outputs = pre_generator(line, hint, False)
        self.pred_model = keras.Model(inputs=[line, hint], outputs=outputs)

        line = keras.Input(shape=(512, 512, 1), dtype=tf.float32, name="input_line_512")
        hint = keras.Input(shape=(512, 512, 3), dtype=tf.float32, name="input_hint_512")
        outputs = generator(line, hint, False)
        self.model = keras.Model(inputs=[line, hint], outputs=outputs)

        self.sess.run(tf.global_variables_initializer())

        self.pred_model.load_weights(self.pred_model_path)
        self.model.load_weights(self.model_path)
        self.sketch_keras = SketchKeras(self.sess)

    def convert2f32(self, img):
        img = img.astype(np.float32)
        return (img / 127.5) - 1.0

    def resize_up(self, imgs):
        resized = [cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC) for img in imgs]
        return np.array(resized)

    def _line_img_prosessing(self, line):
        line = line[:, :, [0]]
        line = self._img_preprocessing(line)
        return line

    def _img_preprocessing(self, img):
        img = np.expand_dims(img, 0)
        img = self.convert2f32(img)
        return img

    def pred_image(self, line, hint):
        outputs_size = 512
        h, w, _ = line.shape
        size = (128, 128)
        l_512 = cv2.resize(line[:, :, [0]], dsize=(outputs_size, outputs_size), interpolation=cv2.INTER_AREA)
        l_128 = cv2.resize(l_512, dsize=size, interpolation=cv2.INTER_AREA)
        l_128 = np.expand_dims(l_128, 2)
        l_512 = np.expand_dims(l_512, 2)

        hint = cv2.resize(hint, dsize=size, interpolation=cv2.INTER_AREA)
        l_128 = self._img_preprocessing(l_128)
        l_512 = self._img_preprocessing(l_512)
        hint = self._img_preprocessing(hint)

        hint[hint == 1.0] = 2.0
        hint = self.pred_model.predict([l_128, hint])

        # pred 512 pix
        img = self.model.predict([l_512, hint])
        img = img[0]
        if w > h:
            rate = w / h
            h = outputs_size
            w = int(rate * outputs_size)
        else:
            rate = h / w
            w = outputs_size
            h = int(rate * outputs_size)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

        return img
