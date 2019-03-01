import cv2
import numpy as np
from scipy import ndimage
import tensorflow as tf


def get_light_map_single(img):
    gray = img
    gray = gray[None]
    gray = gray.transpose((1, 2, 0))
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = gray.reshape((gray.shape[0], gray.shape[1]))
    highPass = gray.astype(int) - blur.astype(int)
    highPass = highPass.astype(np.float)
    highPass = highPass / 128.0
    return highPass


def normalize_pic(img):
    img = img / np.max(img)
    return img


def resize_img_512_3d(img):
    zeros = np.zeros((1, 3, 512, 512), dtype=np.float)
    zeros[0, 0: img.shape[0], 0: img.shape[1], 0: img.shape[2]] = img
    return zeros.transpose((1, 2, 3, 0))


def active_img_denoise(img):
    mat = img.astype(np.float)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    return mat


def active_img(img):
    mat = img.astype(np.float)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    return mat


class SketchKeras():
    __CKPT = "./GUI/src/saved_model/Liner"
    __MODEL_PATH = "./GUI/src/saved_model/Liner/Liner.meta"

    def __init__(self, sess):
        self.sess = sess
        self.__build_model()

    def __build_model(self):
        saver = tf.train.import_meta_graph(self.__MODEL_PATH, clear_devices=True)
        saver.restore(sess=self.sess, save_path=tf.train.latest_checkpoint(self.__CKPT))

        graph = tf.get_default_graph()
        self.ph_input = graph.get_tensor_by_name("input_1:0")
        self.model = graph.get_tensor_by_name("conv2d_18/BiasAdd:0")

    def get_line(self,image):
        width = float(image.shape[1])
        height = float(image.shape[0])

        from_mat = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        from_mat = from_mat.transpose((2, 0, 1))
        light_map = np.zeros(from_mat.shape, dtype=np.float)

        for channel in range(3):
            light_map[channel] = get_light_map_single(from_mat[channel])

        light_map = normalize_pic(light_map)
        light_map = resize_img_512_3d(light_map)
        outputs = list()
        # batch size 1
        for index in range(3):
            feed = np.expand_dims(light_map[index], 0)
            out = self.sess.run(self.model, feed_dict={self.ph_input: feed})
            outputs.append(out[0])

        line_mat = np.array(outputs)
        line_mat = line_mat.transpose((3, 1, 2, 0))[0]
        line_mat = np.amax(line_mat, 2)
        img = active_img_denoise(line_mat)

        if (width > height):
            rate = width / height
            new_height = 512
            new_width = int(512 * rate)

        else:
            rate = height / width
            new_width = 512
            new_height = int(rate * 512)

        img = cv2.resize(img, (new_width, new_height), cv2.INTER_LINEAR_EXACT)
        img = np.expand_dims(img, 3)
        line = np.concatenate([img, img, img], axis=2)
        return line


    def get_line_from_file(self, path):
        from_mat = cv2.imread(path)
        line = self.get_line(from_mat)
        return line
