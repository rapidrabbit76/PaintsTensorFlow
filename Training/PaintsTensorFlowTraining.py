import os
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

import tensorlayer as tl
import numpy as np
from tqdm import tqdm
from dataset.Datasets import Datasets_512
from model.v1_1_0_b.PaintsTensorFlow import Generator_Draft
from model.v1.PaintsTensorFlow import Generator
import utils
import hyperparameter as hp


class PaintsTensorFlowTrain:
    def __init__(self, modelName="PaintsTensorFlow"):
        self.data_sets = Datasets_512(batch_size=hp.batch_size)
        self.model_name = "{}_resize:{}".format(modelName, 512)
        utils.initdir(self.model_name)

        self.global_steps = tf.train.get_or_create_global_step()
        self.epochs = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.ckpt_path = "./ckpt/{}/".format(self.model_name) + "ckpt_E:{}"
        self.ckpt_prefix = os.path.join(self.ckpt_path, "model_GS:{}")

        self.generator_128 = Generator_Draft(resize=True)
        self.generator_512 = Generator(convertUint8=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.5, beta2=0.9)

        self.log_writer = tf.contrib.summary.create_file_writer("./ckpt/{}/board/log".format(self.model_name))
        self.log_writer.set_as_default()
        self.check_point = tf.train.Checkpoint(generator_512=self.generator_512,
                                               optimizer=self.optimizer,
                                               globalSteps=self.global_steps,
                                               epochs=self.epochs)
        self.__load_weights()

    def __load_weights(self):
        zero_1 = np.zeros(shape=[1, 128, 128, 1], dtype=np.float32)
        zero_2 = np.zeros(shape=[1, 128, 128, 3], dtype=np.float32)
        self.generator_128(zero_1, zero_2, False)
        self.generator_128.load_weights("./ckpt/PaintsTensorFlowDraftModel_resize:128/generator_128.h5")

    def __loging(self, name, scalar):
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar(name, scalar)

    def __loss(self, output, target):
        loss = tf.reduce_mean(tf.abs(target - output))
        return loss

    def __pred_image(self, model, image, line, hint, pred, epoch=None):
        gs = self.global_steps.numpy()
        predImage = model(line, pred, hint, training=False)
        file_name = "./ckpt/{}/image/{}.jpg".format(self.model_name, gs)

        if epoch is not None:
            loss = self.__loss(predImage, image)
            self.__loging("Sample_LOSS", loss)
            loss = "{:0.05f}".format(loss).zfill(7)
            print("Epoch:{} GS:{} LOSS:{}".format(epoch, self.global_steps.numpy(), loss))
            file_name = "./ckpt/{}/image/{}_loss:{}.jpg".format(self.model_name, gs, loss)

        hint = np.array(hint)
        hint[hint > 1] = 1

        lineImage = np.concatenate([line, line, line], -1)
        save_img = np.concatenate([lineImage, hint, pred, predImage, image], 1)
        save_img = utils.convert2uint8(save_img)
        tl.visualize.save_images(save_img, [1, save_img.shape[0]], file_name)

    def training(self, loadEpochs=0):
        train_sets, test_sets = self.data_sets.buildDataSets()
        log = self.__loging

        self.check_point.restore(tf.train.latest_checkpoint(self.ckpt_path.format(loadEpochs)))

        if self.global_steps.numpy() == 0:
            self.check_point.save(file_prefix=self.ckpt_prefix.format(0, 0))
            print("------------------------------SAVE_INIT-------------------------------------")

        for epoch in range(hp.epoch):
            print("GS: ", self.global_steps.numpy())

            for line_128, hint_128, image, line, hint in tqdm(train_sets, total=hp.batch_steps):
                draft = self.generator_128(line_128, hint_128, training=False)

                with tf.GradientTape() as tape:
                    genOut = self.generator_512(line, draft, hint, training=True)
                    loss = self.__loss(genOut, image)

                # Training
                gradients = tape.gradient(loss, self.generator_512.variables)
                self.optimizer.apply_gradients(zip(gradients, self.generator_512.variables),
                                               global_step=self.global_steps)
                # Loging
                gs = self.global_steps.numpy()
                if gs % hp.log_interval == 0:
                    log("LOSS", loss)
                    if gs % hp.sampling_interval == 0:
                        # test image Save
                        for line_128, hint_128, image, line, hint in test_sets.take(1):
                            pred = self.generator_128(line_128, hint_128, training=False)
                            self.__pred_image(self.generator_512, image, line, hint, pred, self.epochs.numpy())

                    if gs % hp.save_interval == 0:
                        self.check_point.save(file_prefix=self.ckpt_prefix.format(self.epochs.numpy(), gs))
                        print("------------------------------SAVE_E:{}_G:{}-------------------------------------"
                              .format(self.epochs.numpy(), gs))

            self.check_point.save(file_prefix=self.ckpt_prefix.format(self.epochs.numpy(), self.global_steps.numpy()))
            self.epochs = self.epochs + 1

        for line_128, hint_128, image, line, hint in test_sets.take(1):
            hint = hint.numpy()
            pred = self.generator_128(line_128, hint_128, training=False)
            self.__pred_image(self.generator_512, image, line, hint, pred, self.epochs.numpy())

        ckpt = tf.train.Checkpoint(generator_128=self.generator_128,
                                   generator_512=self.generator_512)
        ckpt.save(file_prefix="./ckpt/" + self.model_name + "/generator")
        self.check_point.save(file_prefix=self.ckpt_prefix.format("Done", self.global_steps.numpy()))

        self.generator_128.save_weights("./ckpt/" + self.model_name + "/PredPaintsTensorFlow.h5")
        self.generator_512.save_weights("./ckpt/" + self.model_name + "/PaintsTensorFlow.h5")

        print("------------------------------Training Done-------------------------------------")
