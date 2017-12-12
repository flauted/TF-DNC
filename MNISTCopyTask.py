#!/usr/bin/env python
"""Repeat a sequence of MNIST images.

The goal is to replicate a sequence of MNIST images. The input is a sequence
of MNIST images, followed by black images of the same sequence length.
The ideal output is a vector of zeros for each actual image. When the
black images begin, the DNC should emit a one-hot vector repeating the
class of the real images.

"""
import numpy as np
import tensorflow as tf
import argparse
import sys
import os
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug
from DNCv3 import DNC
from DNCTrainOps import masked_xent, RMS_train, state_update


class ConvModel:
    """A simple convolutional neural network to be used as a controller.

    Attributes:
        input_size
        output_size
        read_vec_size
        batch_size
        weights: A list of all weights in the network (for regulated loss).
        biases: A list of all biases in the network (for regulated loss).

    """

    def __init__(self, input_size, output_size, read_vec_size, batch_size):  # NOQA
        self.input_size = input_size
        self.output_size = output_size
        self.read_vec_size = read_vec_size
        self.batch_size = batch_size
        self.weights = []
        self.biases = []

    def conv_layer(
            self, inputs, filters_out, kernel_shape=(5, 5), stride=(2, 2)):
        """A basic convolutional layer with VALID padding and relu activation.

        Args:
            inputs: The NCHW stack of images.
            filters_out: The desired number of output filters.

        Keyword Args:
            kernel_shape (``(5, 5)``): A length-2 iterable.
            stride (``(2, 2)``): A length-2 iterable.

        """
        filters_in = inputs.get_shape().as_list()[1]
        K = tf.get_variable(
            "Weights",
            shape=[kernel_shape[0],
                   kernel_shape[1],
                   filters_in,
                   filters_out],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.weights.append(K)
        b = tf.get_variable(
            "Bias",
            shape=[1, filters_out, 1, 1],
            initializer=tf.zeros_initializer())
        self.biases.append(b)
        conv = tf.nn.conv2d(
            inputs,
            K,
            strides=(1, 1, stride[0], stride[1]),
            padding="VALID",
            data_format="NCHW")
        z = conv + b
        a = tf.nn.relu(z)
        return a

    def __call__(self, inputs):
        """Control the DNC."""
        imgs = tf.reshape(
            inputs[:, :28*28],
            [self.batch_size, 1, 28, 28])
        read_vecs = inputs[:, 28*28:]
        with tf.variable_scope("Conv1"):
            a1 = self.conv_layer(imgs, 32, kernel_shape=(5, 5), stride=(2, 2))
        with tf.variable_scope("Conv2"):
            a2 = self.conv_layer(a1, 64, kernel_shape=(5, 5), stride=(2, 2))
        with tf.variable_scope("FC1"):
            conv_out_shape = a2.get_shape().as_list()
            feats = np.prod(conv_out_shape[1:])
            conv_out = tf.reshape(a2, [self.batch_size, feats])
            fc1_input = tf.concat([conv_out, read_vecs], axis=1)
            W = tf.get_variable(
                "Weights",
                shape=[feats+self.read_vec_size, self.output_size],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.weights.append(W)
            b = tf.get_variable(
                "Bias",
                shape=[self.output_size],
                initializer=tf.zeros_initializer())
            self.biases.append(b)
            fc1_act = tf.nn.relu(tf.matmul(fc1_input, W) + b)
            output = fc1_act
        return output


def builtin_mnist_input(mnist_dset, seq_len, batch_size, train=True):
    """Use the MNIST input module from the tutorials.

    Before, create an mnist dataset using the tutorial api. Call this during
    the training loop.

    """
    target_seq_batch = []
    input_seq_batch = []
    for _ in range(seq_len):
        if train:
            img_batch, lbl_batch = mnist_dset.train.next_batch(batch_size)
        else:
            img_batch, lbl_batch = mnist_dset.test.next_batch(batch_size)
        input_seq_batch.append(img_batch)
        target_seq_batch.append(lbl_batch)
    i_data = np.stack(input_seq_batch, 1)
    o_data = np.stack(target_seq_batch, 1)
    final_i_data = np.concatenate(
        (i_data, np.zeros([batch_size, seq_len, 28*28])), axis=1)
    final_o_data = np.concatenate(
        (np.zeros([batch_size, seq_len, 10]), o_data), axis=1)
    return final_o_data, final_i_data


def run_training(seq_len=3,
                 seq_width=10,
                 iterations=50000,
                 mem_len=6,
                 bit_len=1024,
                 num_read_heads=3,
                 num_write_heads=2,
                 batch_size=25,
                 softmax_alloc=False,
                 stateful=False,
                 download_path="/tmp/tensorflow/mnist/input_data",
                 tb_dir_train="tb/dnc/train",
                 tb_dir_test="tb/dnc/test"):
    """Run training loop."""
    print("loading mnist data: ")
    mnist = input_data.read_data_sets(download_path, one_hot=True)
    print("Done")
    graph = tf.Graph()

    with graph.as_default():
        with tf.Session() as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            i_data = tf.placeholder(tf.float32,
                                    [batch_size, seq_len*2, 28*28])
            o_data = tf.placeholder(tf.float32,
                                    [batch_size, seq_len*2, seq_width])

            dnc = DNC(
                input_size=28*28,
                output_size=seq_width,
                seq_len=seq_len,
                mem_len=mem_len,
                bit_len=bit_len,
                batch_size=batch_size,
                n_read_heads=num_read_heads,
                n_write_heads=num_write_heads,
                softmax_allocation=softmax_alloc)
            dnc.install_controller(
                ConvModel(
                    dnc.nn_input_size,
                    dnc.nn_output_size,
                    bit_len*num_read_heads,
                    batch_size))
            initial_state = dnc.zero_state()
            output, new_state = tf.nn.dynamic_rnn(
                dnc,
                i_data,
                initial_state=initial_state,
                time_major=False,
                scope="DNC",
                parallel_iterations=1)

            update_if_stateful = state_update(
                initial_state, new_state, stateful=stateful)
            loss = masked_xent(seq_len=seq_len, seq_width=seq_width,
                               labels=o_data, logits=output)
            tf.summary.scalar("masked_xent", loss)
            summary_op = tf.summary.merge_all()
            apply_gradients = RMS_train(loss)
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(
                tb_dir_train, graph=tf.get_default_graph())
            test_writer = tf.summary.FileWriter(
                tb_dir_test, graph=tf.get_default_graph())

            for epoch in range(iterations+1):
                test = epoch % 100 == 0
                train_summarize = epoch % 100 == 0
                train_o_data, train_i_data = builtin_mnist_input(
                    mnist, seq_len, batch_size, train=True)
                feed_dict = {i_data: train_i_data, o_data: train_o_data}
                train_loss, train_pred, _, _, train_summary = sess.run(
                    [loss, 
                     output, 
                     apply_gradients, 
                     update_if_stateful, 
                     summary_op],
                    feed_dict=feed_dict)
                if train_summarize:
                    train_writer.add_summary(train_summary, epoch)
                    print("[TR]: Epoch [{}], Loss [{}]".format(
                        epoch, train_loss))
                if test:
                    test_o_data, test_i_data = builtin_mnist_input(
                        mnist, seq_len, batch_size, train=False)
                    feed_dict = {i_data: test_i_data, o_data: test_o_data}
                    test_loss, test_pred, test_summary = sess.run(
                        [loss, output, summary_op], feed_dict=feed_dict)
                    test_writer.add_summary(test_summary, epoch)
                    print("[TE]: Epoch [{}]: Loss [{}]".format(
                        epoch, test_loss))

            train_writer.close()
            test_writer.close()

    print(test_i_data)
    print("Last test targets:")
    print(test_o_data)
    print("Last test predictions:")
    print(test_pred)


def main(_):
    run_training(seq_width=10,
                 iterations=FLAGS.epochs,
                 mem_len=FLAGS.mem_len,
                 bit_len=FLAGS.bit_len,
                 num_read_heads=FLAGS.num_read_heads,
                 num_write_heads=FLAGS.num_write_heads,
                 batch_size=FLAGS.batch_size,
                 softmax_alloc=FLAGS.softmax,
                 stateful=FLAGS.stateful,
                 download_path=FLAGS.data_dir,
                 tb_dir_train=os.path.join(FLAGS.tb_dir, FLAGS.tb_train),
                 tb_dir_test=os.path.join(FLAGS.tb_dir, FLAGS.tb_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-e", "--epochs", type=int, default=30000,
        help="Number of epochs (minus one) model trains.")
    parser.add_argument(
        "-tb", "--tb_dir", type=str,
        default="tb/dnc",
        help="Path for folder containing TensorBoard data.")
    parser.add_argument(
        "--tb_train", type=str, default="MNIST_train",
        help="TensorBoard extension for training summary.")
    parser.add_argument(
        "--tb_test", type=str, default="MNIST_test",
        help="TensorBoard extension for test summary.")
    parser.add_argument(
        "-d", "--data_dir", type=str,
        default="/tmp/tensorflow/mnist/input_data",
        help=("-bi false: path to image, label folder. -bi true: path to "
              "save downloaded inputs"))
    parser.add_argument(
        "-s", "--seq_len", type=int, default=3,
        help="Length of (nonzero) input sequence.")
    parser.add_argument(
        "-b", "--batch_size", type=int, default=50,
        help="Number of sequences per step.")
    parser.add_argument(
        "-RH", "--num_read_heads", type=int, default=3)
    parser.add_argument(
        "-WH", "--num_write_heads", type=int, default=2)
    parser.add_argument(
        "-sf", "--stateful", action='store_true', default=False,
        help="Restore state at each train step.")
    parser.add_argument(
        "-sm", "--softmax", action='store_true', default=False,
        help="Use alternative softmax allocation.")
    parser.add_argument(
        "-W", "--bit_len", type=int, default=1024,
        help="Length of a slot in memory.")
    parser.add_argument(
        "-N", "--mem_len", type=int, default=6,
        help="Slots in memory.")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
