from __future__ import division
import os
import sys
import shutil
import math
import tensorflow as tf
from PIL import Image

from HoloGAN.tools.ops import *
from HoloGAN.model_HoloGAN import HoloGAN


AZIMUTH_LOW = -math.pi
AZIMUTH_HIGH = math.pi
AZIMUTH_RANGE = AZIMUTH_HIGH - AZIMUTH_LOW

ELEVATION_LOW = -math.pi / 2
ELEVATION_HIGH = math.pi / 2
ELEVATION_RANGE = ELEVATION_HIGH - ELEVATION_LOW


class ViewHoloGAN(HoloGAN):
    def __init__(self, cfg, sess, input_height=108, input_width=108, crop=True,
            output_height=64, output_width=64,
            gf_dim=64, df_dim=64,
            c_dim=3, dataset_name='lsun',
            input_fname_pattern='*.webp'):

        super().__init__(sess, input_height=input_height,
            input_width=input_width, crop=crop, output_height=output_height,
            output_width=output_width, gf_dim=gf_dim, df_dim=df_dim, c_dim=c_dim,
            dataset_name=dataset_name, input_fname_pattern=input_fname_pattern)

        self.IMAGE_PATH = cfg['image_path']
        self.OUTPUT_DIR = cfg['output_dir']
        # Number of discrete angles to model within both elevation and azimuth ranges
        # The view distribution logits are a NUM_ANGLES x NUM_ANGLES matrix, with
        # the row index -> elevation value, column index -> azimuth value.
        self.NUM_ANGLES = cfg['num_angles']
        self.LOGDIR = os.path.join(self.OUTPUT_DIR, "log")
        self.cfg = cfg


    def build_ViewHoloGAN(self):
        self.inputs = tf.placeholder(tf.float32, [None, self.output_height, self.output_width, self.c_dim], name='real_images')
        self.z = tf.placeholder(tf.float32, [None, self.cfg['z_dim']], name='z')
        inputs = self.inputs

        gen_func = eval("self." + (self.cfg['generator']))
        dis_func = eval("self." + (self.cfg['discriminator']))

        self.view_G, self.view_logits = self.view_generator(self.z)
        self.view_in = tf.placeholder_with_default(self.view_G, [None, 6], name='view_in')
        self.G = gen_func(self.z, self.view_in)

        if str.lower(str(self.cfg["style_disc"])) == "true":
            print("Style Disc")
            self.D, self.D_logits, _, self.d_h1_r, self.d_h2_r, self.d_h3_r, self.d_h4_r = dis_func(inputs, cont_dim=self.cfg['z_dim'], reuse=False)
            self.D_, self.D_logits_, self.Q_c_given_x, self.d_h1_f, self.d_h2_f, self.d_h3_f, self.d_h4_f = dis_func(self.G, cont_dim=self.cfg['z_dim'], reuse=True)

            self.d_h1_loss = self.cfg["DStyle_lambda"] * (
                      tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_r, tf.ones_like(self.d_h1_r))) \
                      + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_f, tf.zeros_like(self.d_h1_f))))
            self.d_h2_loss = self.cfg["DStyle_lambda"] * (
                      tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_r, tf.ones_like(self.d_h2_r))) \
                      + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_f, tf.zeros_like(self.d_h2_f))))
            self.d_h3_loss = self.cfg["DStyle_lambda"] * (
                      tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_r, tf.ones_like(self.d_h3_r))) \
                      + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_f, tf.zeros_like(self.d_h3_f))))
            self.d_h4_loss = self.cfg["DStyle_lambda"] * (
                      tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_r, tf.ones_like(self.d_h4_r))) \
                      + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_f, tf.zeros_like(self.d_h4_f))))
        else:
            self.D, self.D_logits, _, _ = dis_func(inputs, cont_dim=self.cfg['z_dim'], reuse=False)
            self.D_, self.D_logits_, self.Q_c_given_x, self.Pose_c_given_x = dis_func(self.G, cont_dim=self.cfg['z_dim'], reuse=True)


        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        if str.lower(str(self.cfg["style_disc"])) == "true":
            print("Style disc")
            self.d_loss = self.d_loss + self.d_h1_loss + self.d_h2_loss + self.d_h3_loss + self.d_h4_loss
        #====================================================================================================================
        #Identity loss

        self.q_loss = self.cfg["lambda_latent"] * tf.reduce_mean(tf.square(self.Q_c_given_x - self.z))
        self.d_loss = self.d_loss + self.q_loss
        self.g_loss = self.g_loss + self.q_loss

        # Pose loss

        self.pose_loss = self.cfg["lambda_pose"] * tf.reduce_mean(tf.square(self.Pose_c_given_x - self.view_in[:, :2]))
        self.d_loss = self.d_loss + self.pose_loss
        self.g_loss = self.g_loss + self.pose_loss

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def train_ViewHoloGAN(self, config):
        self.d_lr_in = tf.placeholder(tf.float32, None, name='d_eta')
        self.g_lr_in = tf.placeholder(tf.float32, None, name='d_eta')

        d_optim = tf.train.AdamOptimizer(self.cfg['d_eta'], beta1=self.cfg['beta1'], beta2=self.cfg['beta2']).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.cfg['g_eta'], beta1=self.cfg['beta1'], beta2=self.cfg['beta2']).minimize(self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        shutil.copyfile(sys.argv[1], os.path.join(self.LOGDIR, 'config.json'))
        self.g_sum = merge_summary([self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter(self.LOGDIR, self.sess.graph)

        # Sample noise Z and view parameters to test during training
        sample_z = self.sampling_Z(self.cfg['z_dim'], str(self.cfg['sample_z']))
        sample_files = self.data[0:self.cfg['batch_size']]

        if config.dataset == "cats" or config.dataset == "cars":
            sample_images = [get_image(sample_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=False) for sample_file in sample_files]
        else:
            sample_images = [get_image(sample_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=True) for sample_file in sample_files]

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        self.data = glob.glob(os.path.join(self.IMAGE_PATH, self.input_fname_pattern))
        d_lr = self.cfg['d_eta']
        g_lr = self.cfg['g_eta']
        for epoch in range(self.cfg['max_epochs']):
            d_lr = d_lr if epoch < self.cfg['epoch_step'] else d_lr * (self.cfg['max_epochs'] - epoch) / (self.cfg['max_epochs'] - self.cfg['epoch_step'])
            g_lr = g_lr if epoch < self.cfg['epoch_step'] else g_lr * (self.cfg['max_epochs'] - epoch) / (self.cfg['max_epochs'] - self.cfg['epoch_step'])

            random.shuffle(self.data)
            batch_idxs = min(len(self.data), config.train_size) // self.cfg['batch_size']

            for idx in range(0, batch_idxs):
                batch_files = self.data[idx * self.cfg['batch_size']:(idx + 1) * self.cfg['batch_size']]
                if config.dataset == "cats" or config.dataset == "cars":
                    batch_images = [get_image(batch_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=False) for batch_file in batch_files]
                else:
                    batch_images = [get_image(batch_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=self.crop) for batch_file in batch_files]

                batch_z = self.sampling_Z(self.cfg['z_dim'], str(self.cfg['sample_z']))

                feed = {self.inputs: batch_images,
                      self.z: batch_z,
                      self.d_lr_in: d_lr,
                      self.g_lr_in: g_lr}
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],feed_dict=feed)
                self.writer.add_summary(summary_str, counter)
                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict=feed)
                self.writer.add_summary(summary_str, counter)
                # Run g_optim twice
                _, summary_str = self.sess.run([g_optim, self.g_sum],  feed_dict=feed)
                self.writer.add_summary(summary_str, counter)

                counter += 1

                if idx % 250 == 0:
                    errD_fake = self.d_loss_fake.eval(feed)
                    errD_real = self.d_loss_real.eval(feed)
                    errG = self.g_loss.eval(feed)
                    errQ = self.q_loss.eval(feed)
                    errPose = self.pose_loss.eval(feed)
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, q_loss: %.8f, pose_loss: %.8f" \
                        % (epoch, idx, batch_idxs,
                           time.time() - start_time, errD_fake + errD_real, errG, errQ, errPose))

                if np.mod(counter, 500) == 1:
                    self.save(self.LOGDIR, counter)
                    feed_eval = {self.inputs: sample_images,
                               self.z: sample_z,
                               self.d_lr_in: d_lr,
                               self.g_lr_in: g_lr}
                    samples, d_loss, g_loss = self.sess.run(
                      [self.G, self.d_loss, self.g_loss],
                      feed_dict=feed_eval)
                    ren_img = inverse_transform(samples)
                    ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
                    try:
                        tiled = Image.fromarray(merge(ren_img, [self.cfg['batch_size'] // 4, 4]))
                        tiled.save(os.path.join(self.OUTPUT_DIR, "{0}_GAN.png".format(counter)))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    except:
                        first = Image.fromarray(ren_img[0])
                        first.save(os.path.join(self.OUTPUT_DIR, "{0}_GAN.png".format(counter)))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

    def sample_ViewHoloGAN(self, config):
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return
        SAMPLE_DIR = os.path.join(self.OUTPUT_DIR, "samples")
        if not os.path.exists(SAMPLE_DIR):
            os.makedirs(SAMPLE_DIR)
        sample_z = self.sampling_Z(self.cfg['z_dim'], str(self.cfg['sample_z']))
        if config.rotate_azimuth:
            low  = self.cfg['azi_low']
            high = self.cfg['azi_high']
            step = 10
        elif config.rotate_elevation:
            low  = self.cfg['ele_low']
            high = self.cfg['ele_high']
            step = 5
        else:
            low  = 0
            high = 10
            step = 1

        for i in range(low, high, step):
            if config.rotate_azimuth:
                sample_view = np.tile(
                    np.array([i * math.pi / 180.0, 0 * math.pi / 180.0, 1.0, 0, 0, 0]), (self.cfg['batch_size'], 1))
                feed_eval = {self.z: sample_z,
                            self.view_in: sample_view}
            elif config.rotate_azimuth:
                sample_view = np.tile(
                    np.array([270 * math.pi / 180.0, (90 - i) * math.pi / 180.0, 1.0, 0, 0, 0]), (self.cfg['batch_size'], 1))
                feed_eval = {self.z: sample_z,
                            self.view_in: sample_view}
            else:
                feed_eval = {self.z: sample_z}

            if i == low and cfg.graph_pose_dsitribution:
                pose_prob_grid = tf.reshape(tf.nn.softmax(self.view_logits[0]), (self.NUM_ANGLES, self.NUM_ANGLES))
                samples, pose_dist_sample = self.sess.run([self.G, pose_prob_grid], feed_dict=feed_eval)
                # normalize values
                min = np.min(pose_dist_sample)
                max = np.max(pose_dist_sample)
                if max != min:
                    pose_dist_sample = (pose_dist_sample - min) * (255.0 / (max - min))
                pose_dist_img = Image.fromarray((pose_dist_sample).astype('uint8'),'L')
                pose_dist_img.save(os.path.join(SAMPLE_DIR, "{0}_sample_pose_distribution.png".format(counter)))
            else:
                samples = self.sess.run(self.G, feed_dict=feed_eval)
            ren_img = inverse_transform(samples)
            ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
            try:
                tiled = Image.fromarray(merge(ren_img, [self.cfg['batch_size'] // 4, 4]))
                tiled.save(os.path.join(SAMPLE_DIR, "{0}_samples_{1}.png".format(counter, i)))
            except:
                first = Image.fromarray(ren_img[0])
                first.save(os.path.join(SAMPLE_DIR, "{0}_samples_{1}.png".format(counter, i)))




    def discriminator_IN(self, image,  cont_dim, reuse=False):
        if str(self.cfg["add_D_noise"]) == "true":
            image = image + tf.random_normal(tf.shape(image), stddev=0.02)

        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(instance_norm(conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv'),'d_in1'))
            h2 = lrelu(instance_norm(conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv'),'d_in2'))
            h3 = lrelu(instance_norm(conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv'),'d_in3'))

            #Returning logits to determine whether the images are real or fake
            h4 = linear(slim.flatten(h3), 1, 'd_h4_lin')

            # Recognition network for latent variables has an additional layer
            encoder = lrelu((linear(slim.flatten(h3), 128, 'd_latent')))
            cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

            # Pose estimation head
            pose_encoder = lrelu((linear(slim.flatten(h3), 128, 'd_pose')))
            pose_vars = tf.nn.tanh(linear(pose_encoder, 2, "d_pose_prediction")) * tf.constant(math.pi)

            return tf.nn.sigmoid(h4), h4, tf.nn.tanh(cont_vars), pose_vars

    def view_generator(self, z, reuse=False):
        batch_size, dim = tf.shape(z)[0], tf.shape(z)[1]
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

            h1 = lrelu(linear(z, self.NUM_ANGLES * self.NUM_ANGLES // 64, scope='g_view1_linear'))
            h1_sq = tf.reshape(h1, (batch_size, self.NUM_ANGLES // 8, self.NUM_ANGLES // 8, 1))
            h2 = lrelu(instance_norm(deconv2d(h1_sq, (batch_size, self.NUM_ANGLES // 2 , self.NUM_ANGLES // 2, 4), d_h=4, d_w=4, name='g_view2_deconv2d')))
            h3 = deconv2d(h2, (batch_size, self.NUM_ANGLES, self.NUM_ANGLES, 1), k_h=3, k_w=3, name='g_view3_deconv2d')
            view_dist_logits = h3

            view_sample = tf.random.categorical(tf.reshape(view_dist_logits, (batch_size, -1)), 1)
            elev_sample_ix = tf.cast(view_sample // self.NUM_ANGLES, tf.float32)
            azim_sample_ix = tf.cast(view_sample % self.NUM_ANGLES, tf.float32)
            elev_sample = elev_sample_ix / self.NUM_ANGLES * ELEVATION_RANGE + ELEVATION_LOW
            azim_sample = azim_sample_ix / self.NUM_ANGLES * AZIMUTH_RANGE + AZIMUTH_LOW

            param_sample = tf.concat((azim_sample, elev_sample, tf.ones((batch_size, 1)), tf.zeros((batch_size, 3))), axis=1)

            return param_sample, view_dist_logits

    def save(self, checkpoint_dir, step):
        model_name = "HoloGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(
            self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)
