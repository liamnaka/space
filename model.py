import tensorflow as tf
import math
from HoloGAN.tools.ops import linear, lrelu, deconv2d
from HoloGAN.model_HoloGAN import HoloGAN
from PIL import Image

AZIMUTH_LOW = -math.pi
AZIMUTH_HIGH = math.pi
AZIMUTH_RANGE = AZIMUTH_HIGH - AZIMUTH_LOW

ELEVATION_LOW = -math.pi / 2
ELEVATION_HIGH = math.pi / 2
ELEVATION_RANGE = ELEVATION_HIGH - ELEVATION_LOW

# Number of discrete angles to model within both elevation and azimuth ranges
# The view distribution logits are a NUM_ANGLES x NUM_ANGLES matrix, with
# the row index -> elevation value, column index -> azimuth value.
NUM_ANGLES = 120


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

        self.cfg = cfg


    def build_ViewHoloGAN(self):
        self.view_manual = tf.placeholder(tf.float32, [None, 6], name='view_in')
        self.is_manual_view = tf.placeholder(tf.bool, name='is_manual_view')
        self.inputs = tf.placeholder(tf.float32, [None, self.output_height, self.output_width, self.c_dim], name='real_images')
        self.z = tf.placeholder(tf.float32, [None, self.cfg['z_dim']], name='z')
        inputs = self.inputs

        gen_func = eval("self." + (self.cfg['generator']))
        dis_func = eval("self." + (self.cfg['discriminator']))

        self.view_G, self.view_logits = self.view_generator(self.z)
        self.view_in = tf.cond(self.is_manual_view, self.view_manual, self.view_G)
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
            self.D, self.D_logits, _ = dis_func(inputs, cont_dim=self.cfg['z_dim'], reuse=False)
            self.D_, self.D_logits_, self.Q_c_given_x = dis_func(self.G, cont_dim=self.cfg['z_dim'], reuse=True)


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

        shutil.copyfile(sys.argv[1], os.path.join(LOGDIR, 'config.json'))
        self.g_sum = merge_summary([self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter(LOGDIR, self.sess.graph)

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

        self.data = glob.glob(os.path.join(IMAGE_PATH, self.input_fname_pattern))
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
                      self.is_manual_view: False,
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

                errD_fake = self.d_loss_fake.eval(feed)
                errD_real = self.d_loss_real.eval(feed)
                errG = self.g_loss.eval(feed)
                errQ = self.q_loss.eval(feed)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, q_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                       time.time() - start_time, errD_fake + errD_real, errG, errQ))

                if np.mod(counter, 500) == 1:
                    self.save(LOGDIR, counter)
                    feed_eval = {self.inputs: sample_images,
                               self.z: sample_z,
                               self.is_manual_view: False,
                               self.d_lr_in: d_lr,
                               self.g_lr_in: g_lr}
                    samples, d_loss, g_loss = self.sess.run(
                      [self.G, self.d_loss, self.g_loss],
                      feed_dict=feed_eval)
                    ren_img = inverse_transform(samples)
                    ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
                    try:
                        tiled = Image.fromarray(merge(ren_img, [self.cfg['batch_size'] // 4, 4]))
                        tiled.save(os.path.join(OUTPUT_DIR, "{0}_GAN.png".format(counter)))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    except:
                        first = Image.fromarray(ren_img[0])
                        first.save(os.path.join(OUTPUT_DIR, "{0}_GAN.png".format(counter)))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

    def sample_HoloGAN(self, config):
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return
        SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")
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
                            self.is_manual_view: True,
                            self.view_manual: sample_view}
            elif config.rotate_azimuth:
                sample_view = np.tile(
                    np.array([270 * math.pi / 180.0, (90 - i) * math.pi / 180.0, 1.0, 0, 0, 0]), (self.cfg['batch_size'], 1))
                feed_eval = {self.z: sample_z,
                            self.is_manual_view: True,
                            self.view_manual: sample_view}
            else:
                feed_eval = {self.z: sample_z,
                            self.is_manual_view: False}

            samples = self.sess.run(self.G, feed_dict=feed_eval)
            ren_img = inverse_transform(samples)
            ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
            try:
                scipy.misc.imsave(
                  os.path.join(SAMPLE_DIR, "{0}_samples_{1}.png".format(counter, i)),
                  merge(ren_img, [self.cfg['batch_size'] // 4, 4]))
            except:
                scipy.misc.imsave(
                  os.path.join(SAMPLE_DIR, "{0}_samples_{1}.png".format(counter, i)),
                  ren_img[0])


    def view_generator(self, z, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

            batch_size, dim = tf.shape(z)
            h1 = lrelu(linear(z, NUM_ANGLES * NUM_ANGLES / 64, 'g_view1_linear'))
            h1_sq = tf.reshape(h1, (batch_size, NUM_ANGLES / 8, NUM_ANGLES / 8, 1))
            h2 = lrelu(deconv2d(h1_sq, (batch_size, NUM_ANGLES / 2 , NUM_ANGLES / 2, 4), 'g_view2_deconv2d'))
            h3 = deconv2d(h2, (batch_size, NUM_ANGLES, NUM_ANGLES, 1), 'g_view3_deconv2d')
            view_dist_logits = h3

            view_sample = tf.random.categorical(tf.flatten(view_dist_logits), 1)
            elev_sample_ix = tf.cast(view_sample // NUM_ANGLES, tf.float32)
            azim_sample_ix = tf.cast(view_sample % nunm_angles, tf.float32)
            elev_sample = elev_sample_ix / NUM_ANGLES * ELEVATION_RANGE + ELEVATION_LOW
            azim_sample = azim_sample_ix / NUM_ANGLES * AZIMUTH_RANGE + AZIMUTH_LOW

            param_sample = tf.zeros((batch_size, 6), dtype=tf.float32)
            param_sample[:, 0] += elev_sample
            param_sample[:, 1] += azim_sample

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
