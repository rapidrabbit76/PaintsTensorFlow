import tensorflow as tf
from GUI.QtUtil import *
from GUI.tensorModul.SketchKeras import SketchKeras


class PaintsTensorFlowModul:
    __DRAFT_MODEL_PATH__ = "./GUI/model/PaintsTensorFlow/PaintsTensorFlowDraftModel"
    __MODEL_PATH__ = "./GUI/model/PaintsTensorFlow/PaintsTensorFlowModel"

    def __init__(self, sess=None):
        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.2
            self.sess = tf.Session(config=config)
        else:
            self.sess = sess

        self.__build_model()

    def __build_model(self):
        self.__draft_model = tf.contrib.saved_model.load_keras_model(self.__DRAFT_MODEL_PATH__)
        self.__model = tf.contrib.saved_model.load_keras_model(self.__MODEL_PATH__)
        self.sketch_keras = SketchKeras(self.sess)

    def __convert2f32(self, img):
        img = img.astype(np.float32)
        return (img / 127.5) - 1.0

    def __convert2uint8(self, img):
        img = (img + 1) * 127.5
        return img.astype(np.uint8)

    def __img_preprocessing(self, img):
        img = np.expand_dims(img, 0)
        img = self.__convert2f32(img)
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
        l_128 = self.__img_preprocessing(l_128)
        l_512 = self.__img_preprocessing(l_512)
        hint = self.__img_preprocessing(hint)
        hint[hint == 1.0] = 2.0

        draft = self.__draft_model.predict([l_128, hint])
        draft = tf.image.resize_images(draft, size=(512, 512),
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        draft = self.sess.run(draft)
        img = self.__model.predict([l_512, draft])
        img = self.__convert2uint8(img)[0]

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
