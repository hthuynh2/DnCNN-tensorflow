import time

from utils import *
import scipy

try:
    xrange
except:
    xrange = range

def dncnn(input, is_training=True, output_channels=1):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in xrange(2, 16 + 1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block17'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return input - output


class denoiser(object):
    def __init__(self, sess, input_c_dim=1, sigma=25, batch_size=128):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.sigma = sigma
        # build model
        self.Y = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')  # labels
        self.X = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')  # noisy input images
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.Y_hat = dncnn(self.X, is_training=self.is_training) #Predicted clear img
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y - self.Y_hat)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y_hat, self.Y)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    # def evaluate(self, iter_num, test_data, sample_dir, summary_merged, summary_writer):
    #     # assert test_data value range is 0-255
    #     print("[*] Evaluating...")
    #     psnr_sum = 0
    #     for idx in xrange(len(test_data)):
    #         clean_image = test_data[idx].astype(np.float32) / 255.0
    #         output_clean_image, noisy_image, psnr_summary = self.sess.run(
    #             [self.Y_hat, self.X, summary_merged],
    #             feed_dict={self.Y: clean_image,
    #                        self.is_training: False})
    #         summary_writer.add_summary(psnr_summary, iter_num)
    #         groundtruth = np.clip(test_data[idx], 0, 255).astype('uint8')
    #         noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
    #         outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
    #         # calculate PSNR
    #         psnr = cal_psnr(groundtruth, outputimage)
    #         print("img%d PSNR: %.2f" % (idx + 1, psnr))
    #         psnr_sum += psnr
    #         save_images(os.path.join(sample_dir, 'test%d_%d.png' % (idx + 1, iter_num)),
    #                     groundtruth, noisyimage, outputimage)
    #     avg_psnr = psnr_sum / len(test_data)
    #
    #     print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)

    def denoise(self, data):
        output_clean_image = self.sess.run([self.Y_hat],
                                            feed_dict={self.X: data, self.is_training: False})
        return output_clean_image

    def train(self, input_data, label_data, batch_size, ckpt_dir, epoch, lr):
        # assert data range is between 0 and 1
        numBatch = int(input_data.shape[0] / batch_size)
        # load pretrained model
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        index_array = np.arrange(numBatch)
        for epoch in xrange(start_epoch, epoch):
            np.random.shuffle(index_array)
            for batch_id in index_array:
                batch_input = input_data[0][batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                batch_label = label_data[1][batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                _, loss = self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.Y: batch_label, self.X: batch_input, self.lr: lr,
                                                            self.is_training: True})
                iter_num += 1

                if iter_num % 500 == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                          % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                    self.save(iter_num, ckpt_dir)

        print("[*] Finish training.")
        self.save(iter_num, ckpt_dir)

    def save(self, iter_num, ckpt_dir, model_name='DnCNN-tensorflow'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def predict(self, ckpt_dir, save_dir):
        # init variables
        tf.initialize_all_variables().run()
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        for i in range(1, 4000):
            img_path = get_image_path(False, 64, i)
            img = imread(img_path)
            img /= 255.0
            img = img.reshape(1, img.shape[0], img.shape[1], 1)

            output_clean_image = self.sess.run([self.Y_hat],
                                               feed_dict={self.X: img, self.is_training: False})
            output_clean_image = output_clean_image[0]
            outputimage = np.squeeze(output_clean_image)
            outputimage = np.clip(255 * outputimage, 0, 255).astype('uint8')
            scipy.misc.imsave(os.path.join(save_dir, 'denoised_test_%s.png' % format(i, "05")), outputimage)


    def test(self, test_files, ckpt_dir, save_dir):
        """Test DnCNN"""
        # init variables
        tf.initialize_all_variables().run()
        assert len(test_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
        print("[*] " + 'noise level: ' + str(self.sigma) + " start testing...")
        for idx in xrange(len(test_files)):
            clean_image = load_images(test_files[idx]).astype(np.float32) / 255.0

            output_clean_image, noisy_image = self.sess.run([self.Y_hat, self.X],
                                                            feed_dict={self.Y: clean_image, self.is_training: False})
            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
            save_images(os.path.join(save_dir, 'noisy%d.png' % idx), noisyimage)
            save_images(os.path.join(save_dir, 'denoised%d.png' % idx), outputimage)
        avg_psnr = psnr_sum / len(test_files)
        print("--- Average PSNR %.2f ---" % avg_psnr)
