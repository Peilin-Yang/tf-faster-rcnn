from __future__ import print_function, absolute_import, division

from six.moves import range
import os, sys
import math
import time
import csv
import argparse
import multiprocessing

import numpy as np
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
#from tensorflow.python import debug as tf_debug
import cv2 

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

class BibRecogNetwork(object):
    def __init__(self, batch_size=1):
        self._batch_size = batch_size

        self.img_w = 64
        self.img_h = 64
        self.dims = (self.img_w, self.img_h)
        self.SEED = 66478
        self.num_channels = 3
        self.num_labels = 5

        self.img_blobs = tf.placeholder(tf.float32, shape=[None, self.img_w, self.img_h, self.num_channels])
        self.recog = {}

    def batch_normalize(self, x, n_out, phase_train, scope='bn'):
        """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope(scope):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def get_conv2d(self, name, data, patch, d_in, d_out, stride=[1, 1, 1, 1], pooling_size=None, pooling_stride=None,
                   padding='SAME', batch_norm=False, is_training=True, pool_type="max", keep_prob=1.0):
        with tf.name_scope(str('%s' % name)):
            filters = tf.Variable(tf.truncated_normal([patch, patch, d_in, d_out],
                                  stddev=self.get_conv2d_filters_init_stddev(self.img_w, self.img_h, d_in)),
                                  name=str('%s_filters' % name))
            biases = tf.Variable(tf.zeros([d_out]), name=str('%s_b' % name))
            layer = tf.nn.conv2d(data, filters, stride, padding=padding, name=str('%s_layers' % name))
            layer = layer + biases
            layer = tf.nn.relu(layer)
            if batch_norm:
                layer = self.batch_normalize((layer), d_out, tf.convert_to_tensor(is_training, dtype=tf.bool),
                                        str('%s_bn' % name))
            if pooling_stride is not None and pooling_size is not None:
                if pool_type == "max":
                    layer = tf.nn.max_pool(layer, pooling_size, pooling_stride, padding=padding)
                else:
                    layer = tf.nn.avg_pool(layer, pooling_size, pooling_stride, padding=padding)
            if keep_prob < 1.0:
                layer = tf.nn.dropout(layer, keep_prob=keep_prob)
        return filters, biases, layer

    def get_fc(self, name, data, depth, relu=True, keep_prob=1):
        with tf.name_scope(str('%s' % name)):
            inbound = int(data.get_shape()[1])
            weights = tf.Variable(
                tf.truncated_normal([inbound, depth], stddev=math.sqrt(2.0 / inbound), name=str('%s_w' % name)))
            biases = tf.Variable(tf.zeros([depth]), name=str('%s_b' % name))
            layer = tf.matmul(data, weights) + biases
            if relu is True:
                layer = tf.nn.relu(layer)
            if keep_prob < 1:
                layer = tf.nn.dropout(layer, keep_prob=keep_prob, seed=SEED)
            return weights, biases, layer


    def get_conv2d_filters_init_stddev(self, w, h, d_in):
        # from https://arxiv.org/pdf/1502.01852v1.pdf
        return math.sqrt(2.0 / (w * h * d_in))

    def conv_to_fc(self, conv):
        shape = conv.get_shape().as_list()
        return tf.reshape(conv, [-1, shape[1] * shape[2] * shape[3]])

    def get_features(self, data, keep_prob=0.7, is_training=True):
        with tf.name_scope('layers'):
            # Model.
            w1, b1, conv1 = self.get_conv2d('conv1', data=data, patch=5, d_in=3, d_out=48,
                                       stride=[1, 1, 1, 1], pooling_size=[1, 2, 2, 1], pooling_stride=[1, 2, 2, 1],
                                       padding='SAME', batch_norm=True, is_training=is_training)

            w2, b2, conv2 = self.get_conv2d('conv2', data=conv1, patch=5, d_in=48, d_out=64,
                                       stride=[1, 1, 1, 1], padding='SAME', batch_norm=True, is_training=is_training)

            w3, b3, conv3 = self.get_conv2d('conv3', data=conv2, patch=5, d_in=64, d_out=128,
                                       stride=[1, 1, 1, 1], pooling_size=[1, 2, 2, 1], pooling_stride=[1, 2, 2, 1],
                                       padding='SAME',
                                       batch_norm=True, is_training=is_training, keep_prob=keep_prob)

            w4, b4, conv4 = self.get_conv2d('conv4', data=conv3, patch=5, d_in=128, d_out=160,
                                       stride=[1, 1, 1, 1], padding='SAME', batch_norm=True, is_training=is_training)

            w5, b5, conv5 = self.get_conv2d('conv5', data=conv4, patch=5, d_in=160, d_out=192,
                                       stride=[1, 1, 1, 1], pooling_size=[1, 2, 2, 1], pooling_stride=[1, 2, 2, 1],
                                       padding='SAME',
                                       batch_norm=True, is_training=is_training, keep_prob=keep_prob)

            w6, b6, conv6 = self.get_conv2d('conv6', data=conv5, patch=3, d_in=192, d_out=192,
                                       stride=[1, 1, 1, 1], padding='SAME', batch_norm=True, is_training=is_training)

            w7, b7, conv7 = self.get_conv2d('conv7', data=conv6, patch=3, d_in=192, d_out=192,
                                       stride=[1, 1, 1, 1], pooling_size=[1, 2, 2, 1], pooling_stride=[1, 2, 2, 1],
                                       padding='SAME',
                                       batch_norm=True, is_training=is_training, pool_type="avg", keep_prob=keep_prob)

            w8, b8, conv8 = self.get_conv2d('conv8', data=conv7, patch=3, d_in=192, d_out=192,
                                       stride=[1, 1, 1, 1], padding='SAME', batch_norm=True, is_training=is_training)

            w9, b9, conv9 = self.get_conv2d('conv9', data=conv8, patch=3, d_in=192, d_out=384,
                                       stride=[1, 1, 1, 1], pooling_size=[1, 2, 2, 1], pooling_stride=[1, 2, 2, 1],
                                       padding='SAME',
                                       batch_norm=True, is_training=is_training, pool_type="avg", keep_prob=keep_prob)

            w10, b10, conv10 = self.get_conv2d('conv10', data=conv9, patch=2, d_in=384, d_out=768,
                                          stride=[1, 1, 1, 1], pooling_size=[1, 2, 2, 1], pooling_stride=[1, 2, 2, 1],
                                          padding='SAME',
                                          batch_norm=True, is_training=is_training, pool_type="avg", keep_prob=keep_prob)

            reg_vars = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10]
            regularizers = 0;
            for val in reg_vars:
                regularizers += tf.nn.l2_loss(val)
            return self.conv_to_fc(conv10), regularizers

    def get_logits(self, features):
        with tf.name_scope('logits'):
            # Length logits and weights
            length_weigths, length_biases, logits_length = self.get_fc('logits_L', features, self.num_labels, relu=False)

            # Digits logits and weights
            digits_pack = [self.get_fc(str("logits_D%d" % i), features, 10, relu=False) for i in range(self.num_labels)]
            logits_digits = tf.stack([digits_pack[i][2] for i in range(self.num_labels)])

            #softmax
            softmax_l = tf.nn.softmax(logits_length)
            softmax_d = [tf.nn.softmax(logits_digits[i]) for i in range(self.num_labels)]

            #prediction
            # preds_l = tf.reshape(tf.cast(tf.argmax(softmax_l, 1), tf.int32), [])
            # preds_d = tf.reshape(tf.stack([tf.cast(tf.argmax(softmax_d[i], 1), tf.int32) for i in range(num_labels)]), [-1])
            preds_l = tf.cast(tf.argmax(softmax_l, 1), tf.int32)
            preds_d = tf.stack([tf.cast(tf.argmax(softmax_d[i], 1), tf.int32) for i in range(self.num_labels)])

            return preds_l, preds_d

    def _get_blobs(self, filenames):
        """Converts an image into a network input.

        Arguments:
            im (ndarray): a color image in BGR order

        Returns:
            blob (ndarray): a data blob holding an image pyramid
        """
        blob = np.zeros((len(filenames), self.img_w, self.img_h, 3),
                  dtype=np.float32)
        for i, fn in enumerate(filenames):
            im = cv2.imread(fn)
            # im_orig = im.astype(np.float32, copy=True)
            # im_shape = im_orig.shape
            # if im_shape != (self.img_w, self.img_h):
            resized = cv2.resize(im, self.dims)
            blob[i, 0:self.img_w, 0:self.img_h, :] = resized

        return blob


    def read_images(self, filenames):
        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.WholeFileReader()
        filename, filecontents = reader.read(filename_queue)
        image = tf.image.decode_jpeg(filecontents, channels=3)
        image = tf.image.resize_images(image, dims)
        return filename, image

    def get_inputs(self, files, batch_size=16):
        thread_count = multiprocessing.cpu_count()
        #thread_count = 1
        # The minimum number of instances in a queue from which examples are drawn
        # randomly. The larger this number, the more randomness at the expense of
        # higher memory requirements.
        MIN_AFTER_DEQUEUE = 100

        # When batching data, the queue's capacity will be larger than the batch_size
        # by some factor. The recommended formula is (num_threads + a small safety
        # margin). For now, we use a single thread for reading, so this can be small.
        QUEUE_SIZE_MULTIPLIER = thread_count + 3
        capacity = MIN_AFTER_DEQUEUE + QUEUE_SIZE_MULTIPLIER * batch_size
        # input images
        with tf.name_scope('input'):
            # get single examples
            filenames, images = read_images(files)
            # groups examples into batches randomly
            data = None
            data = tf.train.batch([filenames, images],
                                 batch_size=batch_size,
                                 capacity=capacity,
                                 num_threads=thread_count)
            return data

    def format_results(self, a):
        return '%s' % (''.join(np.char.mod('%d', a[1:a[0]+1])))

    def predict(self, sess, feed_img_blobs):  
        vpreds_d, vpreds_l = sess.run([
            self.recog['logits_digits'], 
            self.recog['logits_length']
            ], feed_dict={self.img_blobs:feed_img_blobs})
        results = np.vstack((vpreds_l, vpreds_d)).T
        # output = np.apply_along_axis(self.format_results, 1, results)
        return results

    def build_network(self):
        self.recog['features'], self.recog['regularizers'] = \
                self.get_features(self.img_blobs, keep_prob=1, is_training=False)
        self.recog['logits_length'], self.recog['logits_digits'] = \
                self.get_logits(self.recog['features'])
                
def imgs_recognition(sess, net, filenames, start, end):
    all_recognitions = [[] for _ in range(end-start)]
    blobs = net._get_blobs(filenames[start:end])
    #print(filenames[start:end])

    _t = Timer()
    _t.tic()
    res = net.predict(sess, blobs)
    tf.logging.info('%d images were recognized. avg time: %.3f'% ((end-start), _t.average_time))
    _t.toc()

    return res


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_id', type=str, default='predict')
    parser.add_argument('--images', type=str, required=True)
    parser.add_argument('--checkpoints_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)
    last_checkpoint = tf.train.latest_checkpoint(os.path.join(args.checkpoints_dir))
    if last_checkpoint is None:
        tf.logging.error("No checkpoint available for model evaluation. Exiting...")
        exit()

    net = BibRecogNetwork(args.batch_size)
    net.build_network()
    saver = tf.train.Saver()
    saver.restore(sess, last_checkpoint)

    tf.logging.info('Starting collecting files...')
    collection_files_start = time.time()
    filenames = []
    if os.path.isdir(args.images):
        for root, dirs, files in os.walk(args.images):
            for name in files:
                filenames.append(os.path.join(root, name))
    elif os.path.isfile(args.images):
        filenames.append(args.images)
    else:
        print('Please provide the valid image paths!! Exit...')
        exit()
    collection_files_end = time.time()
    dataset_size = len(filenames)

    tf.logging.info('Ending collecting files...')
    tf.logging.info('Time Elasped: %.1fs' % (collection_files_end - collection_files_start))
    tf.logging.info('%d files collected' % dataset_size)

    if args.output_path:
        if os.path.isfile(args.output_path):
            os.remove(args.output_path)

    eval_qty = 0
    while eval_qty < dataset_size:
        start = eval_qty
        end = min(eval_qty+args.batch_size, dataset_size)
        res = imgs_recognition(sess, net, filenames, start, end)
        for idx in range(start, end):
            l = res[idx-start].tolist()
            num = ''.join([str(n) for n in l[1:l[0]+2]])
            print('%s %s' % (filenames[idx], num))
        eval_qty += args.batch_size
        
    sess.close()
