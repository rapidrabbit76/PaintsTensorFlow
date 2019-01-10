import os
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

import tensorlayer as tl
import numpy as np
from tqdm import tqdm
from dataset.Datasets import Datasets, Datasets_512
from model.PaintsTensorFlow import *
import utils


class PaintsTensorFlowTrain_128:
    def __init__(self, modelName="PaintsTensorFlow"):
        self.dataSets = Datasets(batch_size=hp.batch_size)
        self.modelName = "{}_resize:{}".format(modelName, 128)
        base = os.path.join("./ckpt", self.modelName)
        utils.mkdir("./ckpt")
        utils.mkdir(base)
        utils.mkdir(os.path.join(base, "board"))
        utils.mkdir(os.path.join(base, "image"))

        self.globalSteps = tf.train.get_or_create_global_step()
        self.genOptimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=hp.beta1, beta2=hp.beta2)
        self.disOptimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=hp.beta1, beta2=hp.beta2)

        self.ckptPath = "./ckpt/{}/".format(self.modelName) + "ckpt_E:{}"
        self.ckptPrefix = os.path.join(self.ckptPath, "model_GS:{}")

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.logWriter = tf.contrib.summary.create_file_writer("./ckpt/{}/board/log".format(self.modelName))
        self.logWriter.set_as_default()

        self.checkPoint = tf.train.Checkpoint(generator=self.generator,
                                              genOptimizer=self.genOptimizer,
                                              disOptimizer=self.disOptimizer,
                                              discriminator=self.discriminator,
                                              globalSteps=self.globalSteps)

    def _discriminator_loss(self, real, fake):
        SCE = tf.losses.sigmoid_cross_entropy

        self.realLoss = SCE(multi_class_labels=tf.ones_like(real), logits=real)
        self.fakeLoss = SCE(multi_class_labels=tf.zeros_like(fake), logits=fake)
        loss = self.realLoss + self.fakeLoss
        return loss

    def _generator_loss(self, disOutput, output, target):
        SCE = tf.losses.sigmoid_cross_entropy
        self.ganLoss = SCE(multi_class_labels=tf.ones_like(disOutput), logits=disOutput)
        self.imageLoss = tf.reduce_mean(tf.abs(target - output)) * hp.l1_scaling
        self.lineLoss = self.line_loss(output, target) * hp.l2_scaling
        loss = self.imageLoss + self.ganLoss + self.lineLoss
        return loss

    def line_loss(self, output, target):
        def convert2uint8(img):
            img = (img + 1) * 127.5
            return tf.cast(img, tf.uint8)

        def convert2f32(img):
            img = tf.cast(img, tf.float32)
            return (img / 127.5) - 1.0

        output = convert2uint8(output)
        target = convert2uint8(target)

        preadLine = utils.get_line(output.numpy())
        realLine = utils.get_line(target.numpy())

        preadLine = convert2f32(preadLine)
        realLine = convert2f32(realLine)

        loss = tf.reduce_mean(tf.abs(realLine - preadLine))
        return loss

    def _contrastLoss(self, generatorOutput):
        loss = 0
        for ch in range(generatorOutput.shape[3]):
            chOut = generatorOutput[:, :, :, ch]
            loss += tf.reduce_mean(tf.square(chOut - tf.reduce_mean(chOut)))
        return loss

    def predImage(self, model, image, line, hint, epoch=None):
        gs = self.globalSteps.numpy()
        predImage = model(line, hint, training=False)

        zeroHint = tf.ones_like(hint)
        zeroHint += 1
        predImageZero = model(line, zeroHint, training=False)

        disFake = self.discriminator(predImage, training=False)
        loss = self._generator_loss(disFake, predImage, image)

        self._loging("Sample_LOSS", loss)
        loss = "{:0.05f}".format(loss).zfill(7)
        print("Epoch:{} GS:{} LOSS:{}".format(epoch, self.globalSteps.numpy(), loss))
        file_name = "./ckpt/{}/image/{}_loss:{}.jpg".format(self.modelName, gs, loss)

        hint = np.array(hint)
        hint[hint > 1] = 1

        lineImage = np.concatenate([line, line, line], -1)
        save_img = np.concatenate([predImageZero, predImage, lineImage, hint, image], 1)
        save_img = utils.convert2uint8(save_img)
        tl.visualize.save_images(save_img, [1, save_img.shape[0]], file_name)

    def _loging(self, name, scalar):
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar(name, scalar)

    def training(self, loadEpochs=0):
        train_sets, test_sets = self.dataSets.buildDataSets()
        log = self._loging

        self.checkPoint.restore(tf.train.latest_checkpoint(self.ckptPath.format(loadEpochs)))

        if self.globalSteps.numpy() == 0:
            self.checkPoint.save(file_prefix=self.ckptPrefix.format(0, 0))
            print("------------------------------SAVE_INIT-------------------------------------")

        for epoch in range(hp.epoch):
            print("GS: ", self.globalSteps.numpy())

            for image, line, hint in tqdm(train_sets, total=hp.batch_steps):
                # get loss
                with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
                    predImage = self.generator(line, hint, training=True)
                    disReal = self.discriminator(image, training=True)
                    disFake = self.discriminator(predImage, training=True)

                    generatorLoss = self._generator_loss(disFake, predImage, image)
                    discriminatorLoss = self._discriminator_loss(disReal, disFake)

                # Training
                discriminator_gradients = discTape.gradient(discriminatorLoss, self.discriminator.variables)
                generator_gradients = genTape.gradient(generatorLoss, self.generator.variables)

                self.disOptimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.variables))
                self.genOptimizer.apply_gradients(zip(generator_gradients, self.generator.variables),
                                                  global_step=self.globalSteps)
                gs = self.globalSteps.numpy()

                if gs % hp.log_interval == 0:
                    log("LOSS_G", generatorLoss)
                    log("LOSS_G_Image", self.imageLoss)
                    log("LOSS_G_GAN", self.ganLoss)
                    log("LOSS_G_LineLoss", self.lineLoss)
                    log("LOSS_D", discriminatorLoss)
                    log("LOSS_D_Real", self.realLoss)
                    log("LOSS_D_Fake", self.fakeLoss)

                    if gs % hp.sampling_interval == 0:
                        for image, line, hint in test_sets.take(1):
                            self.predImage(self.generator, image, line, hint, epoch)

                    if gs % hp.save_interval == 0:
                        self.checkPoint.save(file_prefix=self.ckptPrefix.format(epoch, gs))
                        print("------------------------------SAVE_E:{}_G:{}-------------------------------------"
                              .format(epoch, gs))
            self.checkPoint.save(file_prefix=self.ckptPrefix.format(epoch, gs))

        self.generator.summary()
        ckpt = tf.train.Checkpoint(generator_128=self.generator)
        ckpt.save(file_prefix="./ckpt/" + self.modelName + "/generator_128")
        self.checkPoint.save(file_prefix=self.ckptPrefix.format("Done", gs))
        self.generator.save_weights("./ckpt/" + self.modelName + "/generator_128.h5")
        print("------------------------------Training Done-------------------------------------")


class PaintsTensorFlowTrain_512:
    def __init__(self, modelName="PaintsTensorFlow"):
        self.dataSets = Datasets_512(batch_size=hp.batch_size)
        self.modelName = "{}_resize:{}".format(modelName, 512)
        base = os.path.join("./ckpt", self.modelName)
        utils.mkdir(base)
        utils.mkdir(os.path.join(base, "board"))
        utils.mkdir(os.path.join(base, "image"))

        self.globalSteps = tf.train.get_or_create_global_step()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=hp.beta1, beta2=hp.beta2)

        self.ckptPath = "./ckpt/{}/".format(self.modelName) + "ckpt_E:{}"
        self.ckptPrefix = os.path.join(self.ckptPath, "model_GS:{}")

        self.generator_128 = Generator(resize=True)
        self.generator_512 = Generator()

        self.logWriter = tf.contrib.summary.create_file_writer("./ckpt/{}/board/log".format(self.modelName))
        self.logWriter.set_as_default()
        self.checkPoint = tf.train.Checkpoint(generator_512=self.generator_512,
                                              optimizer=self.optimizer,
                                              globalSteps=self.globalSteps)
        self.__load_weights()

    def __load_weights(self):
        zero_1 = np.zeros(shape=[1, 128, 128, 1], dtype=np.float32)
        zero_2 = np.zeros(shape=[1, 128, 128, 3], dtype=np.float32)
        self.generator_128(zero_1, zero_2, False)
        self.generator_128.load_weights("./ckpt/PaintsTensorFlow_resize:128/generator_128.h5")

    def _loging(self, name, scalar):
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar(name, scalar)

    def loss(self, output, target):
        loss = tf.reduce_mean(tf.abs(target - output))
        return loss

    def predImage(self, model, image, line, hint, pred, epoch=None):
        gs = self.globalSteps.numpy()
        predImage = model(line, pred, training=False)
        file_name = "./ckpt/{}/image/{}.jpg".format(self.modelName, gs)

        if epoch is not None:
            loss = self.loss(predImage, image)
            self._loging("Sample_LOSS", loss)
            loss = "{:0.05f}".format(loss).zfill(7)
            print("Epoch:{} GS:{} LOSS:{}".format(epoch, self.globalSteps.numpy(), loss))
            file_name = "./ckpt/{}/image/{}_loss:{}.jpg".format(self.modelName, gs, loss)

        hint = np.array(hint)
        hint[hint > 1] = 1

        lineImage = np.concatenate([line, line, line], -1)
        save_img = np.concatenate([lineImage, hint, pred, predImage, image], 1)
        save_img = utils.convert2uint8(save_img)
        tl.visualize.save_images(save_img, [1, save_img.shape[0]], file_name)

    def training(self, loadEpochs=0):
        train_sets, test_sets = self.dataSets.buildDataSets()
        log = self._loging

        self.checkPoint.restore(tf.train.latest_checkpoint(self.ckptPath.format(loadEpochs)))

        if self.globalSteps.numpy() == 0:
            self.checkPoint.save(file_prefix=self.ckptPrefix.format(0, 0))
            print("------------------------------SAVE_INIT-------------------------------------")

        for epoch in range(hp.epoch):
            print("GS: ", self.globalSteps.numpy())

            for line_128, hint_128, image, line, hint in tqdm(train_sets, total=hp.batch_steps):
                hint = hint.numpy()
                hint = self.generator_128(line_128, hint_128, training=False)

                with tf.GradientTape() as tape:
                    genOut = self.generator_512(line, hint, training=True)
                    loss = self.loss(genOut, image)

                # Training
                gradients = tape.gradient(loss, self.generator_512.variables)
                self.optimizer.apply_gradients(zip(gradients, self.generator_512.variables),
                                               global_step=self.globalSteps)
                gs = self.globalSteps.numpy()
                if gs % hp.log_interval == 0:
                    log("LOSS", loss)

                    if gs % hp.sampling_interval == 0:
                        # test image Save
                        for line_128, hint_128, image, line, hint in test_sets.take(1):
                            hint = hint.numpy()
                            pred = self.generator_128(line_128, hint_128, training=False)
                            self.predImage(self.generator_512, image, line, hint, pred, epoch)

                    if gs % hp.save_interval == 0:
                        self.checkPoint.save(file_prefix=self.ckptPrefix.format(epoch, gs))
                        print("------------------------------SAVE_E:{}_G:{}-------------------------------------"
                              .format(epoch, gs))

            self.checkPoint.save(file_prefix=self.ckptPrefix.format(epoch, self.globalSteps.numpy()))

        for line_128, hint_128, image, line, hint in test_sets.take(1):
            hint = hint.numpy()
            pred = self.generator_128(line_128, hint_128, training=False)
            self.predImage(self.generator_512, image, line, hint, pred, 100)

        ckpt = tf.train.Checkpoint(generator_128=self.generator_128,
                                   generator_512=self.generator_512)
        ckpt.save(file_prefix="./ckpt/" + self.modelName + "/generator")
        self.checkPoint.save(file_prefix=self.ckptPrefix.format("Done", self.globalSteps.numpy()))

        self.generator_128.save_weights("./ckpt/" + self.modelName + "/PredPaintsTensorFlow.h5")
        self.generator_512.save_weights("./ckpt/" + self.modelName + "/PaintsTensorFlow.h5")

        print("------------------------------Training Done-------------------------------------")