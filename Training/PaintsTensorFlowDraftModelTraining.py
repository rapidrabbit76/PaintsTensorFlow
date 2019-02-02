import os
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

import tensorlayer as tl
import numpy as np
from tqdm import tqdm
from dataset.Datasets import Datasets
from model.v1_1_0_b.PaintsTensorFlow import Generator_Draft, Discriminator
import utils
import hyperparameter as hp


class PaintsTensorFlowDraftModelTrain:
    def __init__(self, modelName="PaintsTensorFlowDraftModel"):
        self.dataSets = Datasets(batch_size=hp.batch_size)
        self.modelName = "{}_resize:{}".format(modelName, 128)
        base = os.path.join("./ckpt", self.modelName)
        utils.mkdir("./ckpt")
        utils.mkdir(base)
        utils.mkdir(os.path.join(base, "board"))
        utils.mkdir(os.path.join(base, "image"))

        self.globalSteps = tf.train.get_or_create_global_step()
        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=hp.beta1, beta2=hp.beta2)
        self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=hp.beta1, beta2=hp.beta2)

        self.ckptPath = "./ckpt/{}/".format(self.modelName) + "ckpt_E:{}"
        self.ckptPrefix = os.path.join(self.ckptPath, "model_GS:{}")

        self.generator = Generator_Draft(name="draft_net")
        self.discriminator = Discriminator()

        self.logWriter = tf.contrib.summary.create_file_writer("./ckpt/{}/board/log".format(self.modelName))
        self.logWriter.set_as_default()

        self.check_point = tf.train.Checkpoint(generator=self.generator,
                                               genOptimizer=self.generator_optimizer,
                                               disOptimizer=self.discriminator_optimizer,
                                               discriminator=self.discriminator,
                                               globalSteps=self.globalSteps)

    def __discriminator_loss(self, real, fake):
        SCE = tf.losses.sigmoid_cross_entropy

        self.real_loss = SCE(multi_class_labels=tf.ones_like(real), logits=real)
        self.fake_loss = SCE(multi_class_labels=tf.zeros_like(fake), logits=fake)
        loss = self.real_loss + self.fake_loss
        return loss

    def __generator_loss(self, disOutput, output, target):
        SCE = tf.losses.sigmoid_cross_entropy
        self.gan_loss = SCE(multi_class_labels=tf.ones_like(disOutput), logits=disOutput)
        self.image_loss = tf.reduce_mean(tf.abs(target - output)) * hp.l1_scaling
        loss = self.image_loss + self.gan_loss
        return loss

    def pred_image(self, model, image, line, hint, epoch=None):
        gs = self.globalSteps.numpy()
        pred_image = model(line, hint, training=False)

        zero_hint = tf.ones_like(hint)
        zero_hint += 1
        pred_image_zero = model(line, zero_hint, training=False)

        dis_fake = self.discriminator(pred_image, training=False)
        loss = self.__generator_loss(dis_fake, pred_image, image)

        self._loging("Sample_LOSS", loss)
        loss = "{:0.05f}".format(loss).zfill(7)
        print("Epoch:{} GS:{} LOSS:{}".format(epoch, self.globalSteps.numpy(), loss))
        file_name = "./ckpt/{}/image/{}_loss:{}.jpg".format(self.modelName, gs, loss)

        hint = np.array(hint)
        hint[hint > 1] = 1

        line_image = np.concatenate([line, line, line], -1)
        save_img = np.concatenate([pred_image_zero, pred_image, line_image, hint, image], 1)
        save_img = utils.convert2uint8(save_img)
        tl.visualize.save_images(save_img, [1, save_img.shape[0]], file_name)

    def _loging(self, name, scalar):
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar(name, scalar)

    def training(self, loadEpochs=0):
        train_sets, test_sets = self.dataSets.buildDataSets()
        log = self._loging

        self.check_point.restore(tf.train.latest_checkpoint(self.ckptPath.format(loadEpochs)))

        if self.globalSteps.numpy() == 0:
            self.check_point.save(file_prefix=self.ckptPrefix.format(0, 0))
            print("------------------------------SAVE_INIT-------------------------------------")

        for epoch in range(hp.epoch):
            print("GS: ", self.globalSteps.numpy())

            for image, line, hint in tqdm(train_sets, total=hp.batch_steps):
                # get loss
                with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
                    pred_image = self.generator(line, hint, training=True)
                    dis_real = self.discriminator(image, training=True)
                    dis_fake = self.discriminator(pred_image, training=True)

                    generator_loss = self.__generator_loss(dis_fake, pred_image, image)
                    discriminator_loss = self.__discriminator_loss(dis_real, dis_fake)

                # Training
                discriminator_gradients = discTape.gradient(discriminator_loss, self.discriminator.variables)
                generator_gradients = genTape.gradient(generator_loss, self.generator.variables)

                self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.variables))
                self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.variables),
                                                         global_step=self.globalSteps)
                gs = self.globalSteps.numpy()

                if gs % hp.log_interval == 0:
                    log("LOSS_G", generator_loss)
                    log("LOSS_G_Image", self.image_loss)
                    log("LOSS_G_GAN", self.gan_loss)
                    log("LOSS_D", discriminator_loss)
                    log("LOSS_D_Real", self.real_loss)
                    log("LOSS_D_Fake", self.fake_loss)

                    if gs % hp.sampling_interval == 0:
                        for image, line, hint in test_sets.take(1):
                            self.pred_image(self.generator, image, line, hint, epoch)

                    if gs % hp.save_interval == 0:
                        self.check_point.save(file_prefix=self.ckptPrefix.format(epoch, gs))
                        print("------------------------------SAVE_E:{}_G:{}-------------------------------------"
                              .format(epoch, gs))
            self.check_point.save(file_prefix=self.ckptPrefix.format(epoch, gs))

        self.generator.summary()
        ckpt = tf.train.Checkpoint(generator_128=self.generator)
        ckpt.save(file_prefix="./ckpt/" + self.modelName + "/generator_128")
        self.check_point.save(file_prefix=self.ckptPrefix.format("Done", gs))
        self.generator.save_weights("./ckpt/" + self.modelName + "/generator_128.h5")
        print("------------------------------Training Done-------------------------------------")
