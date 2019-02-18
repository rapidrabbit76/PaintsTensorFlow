import os
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
import tensorlayer as tl
import numpy as np
from tqdm import tqdm
from dataset.Datasets import Datasets
from model.PaintsTensorFlow import Generator, Discriminator
import utils
import hyperparameter as hp


class PaintsTensorFlowDraftModelTrain:
    def __init__(self, model_name="PaintsTensorFlowDraftModel"):
        self.data_sets = Datasets(batch_size=hp.batch_size)
        self.model_name = model_name
        utils.initdir(self.model_name)

        self.global_steps = tf.train.get_or_create_global_step()
        self.epochs = tf.Variable(0, trainable=False, dtype=tf.int32)

        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=hp.beta1, beta2=hp.beta2)
        self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=hp.beta1, beta2=hp.beta2)
        self.ckptPath = "./ckpt/{}/".format(self.model_name) + "ckpt_E:{}"
        self.ckptPrefix = os.path.join(self.ckptPath, "model_GS:{}")

        self.generator = Generator(name="PaintsTensorFlowDraftNet")
        self.discriminator = Discriminator()

        self.logWriter = tf.contrib.summary.create_file_writer("./ckpt/{}/board/log".format(self.model_name))
        self.logWriter.set_as_default()

        self.check_point = tf.train.Checkpoint(generator=self.generator,
                                               genOptimizer=self.generator_optimizer,
                                               disOptimizer=self.discriminator_optimizer,
                                               discriminator=self.discriminator,
                                               globalSteps=self.global_steps,
                                               epochs=self.epochs)

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

    def __pred_image(self, model, image, line, hint, epoch=None):
        global_steps = self.global_steps.numpy()
        pred_image = model.predict([line, hint])

        zero_hint = tf.ones_like(hint)
        zero_hint += 1
        pred_image_zero = model.predict([line, zero_hint])

        dis_fake = self.discriminator(pred_image, training=False)
        loss = self.__generator_loss(dis_fake, pred_image, image)

        self.__loging("Sample_LOSS", loss)
        loss = "{:0.05f}".format(loss).zfill(7)
        print("Epoch:{} GS:{} LOSS:{}".format(epoch, global_steps, loss))
        file_name = "./ckpt/{}/image/{}_loss:{}.jpg".format(self.model_name, global_steps, loss)

        hint = np.array(hint)
        hint[hint > 1] = 1

        line_image = np.concatenate([line, line, line], -1)
        save_img = np.concatenate([line_image, hint, pred_image_zero, pred_image, image], 1)
        save_img = utils.convert2uint8(save_img)
        tl.visualize.save_images(save_img, [1, save_img.shape[0]], file_name)

    def __loging(self, name, scalar):
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar(name, scalar)

    def __check_point_save(self):
        file_prefix = self.ckptPrefix.format(self.epochs.numpy(), self.global_steps.numpy())
        self.check_point.save(file_prefix=file_prefix)

    def training(self, loadEpochs=0):
        train_sets, test_sets = self.data_sets.buildDataSets()
        log = self.__loging

        self.check_point.restore(tf.train.latest_checkpoint(self.ckptPath.format(loadEpochs)))

        for epoch in range(hp.epoch):
            print("GS: ", self.global_steps.numpy(), "Epochs:  ", self.epochs.numpy())

            for image, line, hint in tqdm(train_sets, total=hp.batch_steps):
                # get loss
                with tf.GradientTape() as genTape, tf.GradientTape() as discTape:

                    pred_image = self.generator(inputs=[line, hint], training=True)

                    dis_real = self.discriminator(inputs=image, training=True)
                    dis_fake = self.discriminator(inputs=pred_image, training=True)

                    generator_loss = self.__generator_loss(dis_fake, pred_image, image)
                    discriminator_loss = self.__discriminator_loss(dis_real, dis_fake)

                # Gradients
                discriminator_gradients = discTape.gradient(discriminator_loss, self.discriminator.variables)
                generator_gradients = genTape.gradient(generator_loss, self.generator.variables)

                self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.variables))
                self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.variables),
                                                         global_step=self.global_steps)
                gs = self.global_steps.numpy()

                if gs % hp.log_interval == 0:
                    log("LOSS_G", generator_loss)
                    log("LOSS_G_Image", self.image_loss)
                    log("LOSS_G_GAN", self.gan_loss)
                    log("LOSS_D", discriminator_loss)
                    log("LOSS_D_Real", self.real_loss)
                    log("LOSS_D_Fake", self.fake_loss)

                    if gs % hp.sampling_interval == 0:
                        for image, line, hint in test_sets.take(1):
                            self.__pred_image(self.generator, image, line, hint, self.epochs.numpy())

                    if gs % hp.save_interval == 0:
                        self.__check_point_save()
                        print("------------------------------SAVE_E:{}_G:{}-------------------------------------"
                              .format(self.epochs.numpy(), gs))
            self.epochs = self.epochs + 1

        self.generator.summary()
        save_path = "./ckpt/" + self.model_name + "/{}.h5".format(self.generator.name)
        self.generator.save(save_path, include_optimizer=False)  # for keras Model
        save_path = tf.contrib.saved_model.save_keras_model(self.generator, "./saved_model")  # saved_model
        print("saved_model path = {}".format(save_path))

        print("------------------------------Training Done-------------------------------------")
