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
import struct
import argparse
import sys
import os
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug
from DNCv3 import DNC
from DNCTrainOps import masked_xent, RMS_train, state_update


class ConvModel:
    """A simple convolutional neural network to be used as a controller."""

    def __init__(self, input_size, output_size, read_vec_size, batch_size):  # NOQA
        self.input_size = input_size
        self.output_size = output_size
        self.read_vec_size = read_vec_size
        self.batch_size = batch_size

    def __call__(self, inputs):
        """Control the DNC."""
        input_imgs = tf.reshape(
            inputs[:, :28*28],
            [self.batch_size, 1, 28, 28])
        read_vecs = inputs[:, 28*28:]
        with tf.variable_scope("L1"):
            K1 = tf.get_variable(
                "layer1_weights",
                shape=[5, 5, 1, 32],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b1 = tf.get_variable(
                "layer1_bias",
                shape=[1, 32, 1, 1],
                initializer=tf.zeros_initializer())
            l1_conv = tf.nn.conv2d(
                input_imgs,
                K1,
                strides=(1, 1, 2, 2),
                padding="VALID",
                data_format="NCHW")
            l1_evidence = l1_conv + b1
            l1_act = tf.nn.relu(l1_evidence)
            self.K1 = K1
            self.b1 = b1
        with tf.variable_scope("L2"):
            K2 = tf.get_variable(
                "layer2_weights",
                shape=[5, 5, 32, 64],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b2 = tf.get_variable(
                "layer2_bias",
                shape=[1, 64, 1, 1],
                initializer=tf.zeros_initializer())
            l2_conv = tf.nn.conv2d(
                l1_act,
                K2,
                strides=(1, 1, 2, 2),
                padding="VALID",
                data_format="NCHW")
            l2_act = tf.nn.relu(l2_conv + b2)
            self.K2 = K2
            self.b2 = b2
        with tf.variable_scope("FC1"):
            conv_out_shape = l2_act.get_shape().as_list()
            feats = np.prod(conv_out_shape[1:])
            conv_out = tf.reshape(
                l2_act,
                [self.batch_size, feats])
            fc1_input = tf.concat([conv_out, read_vecs], axis=1)
            W = tf.get_variable(
                "fc1_weights",
                shape=[feats+self.read_vec_size, self.output_size],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(
                "fc1_bias",
                shape=[self.output_size],
                initializer=tf.zeros_initializer())
            fc1_act = tf.nn.relu(tf.matmul(fc1_input, W) + b)
            self.W = W
            self.b = b
            output = fc1_act
        return output


def disk_data(image_path, label_path):
    """Generate inputs for DNC sequence copy task."""
    with open(label_path, 'rb') as f_label:
        magic, num = struct.unpack(">II", f_label.read(8))
        label = np.fromfile(f_label, dtype=np.int8)

    with open(image_path, 'rb') as f_image:
        magic, num, rows, cols = struct.unpack(">IIII", f_image.read(16))
        image = np.fromfile(
            f_image, dtype=np.int8).reshape(len(label), rows, cols)

    get_img = lambda idx: (label[idx], image[idx])  # NOQA
    for i in range(len(label)):
        yield get_img(i)


def _gen_sequence(label_img_gen, seq_len, batch_size):
    """Generate the input and output sequences."""
    target_batch = []
    input_batch = []
    for _ in range(batch_size):
        img_seq = []
        labels = np.zeros([seq_len, 10])
        for timestep in range(seq_len):
            label, img = next(label_img_gen)
            labels[timestep, label] = 1
            img_seq.append(img)
        targets = np.concatenate((np.zeros([seq_len, 10]), labels), axis=0)
        inputs = np.concatenate(
            (np.asarray(img_seq),
             np.zeros([seq_len, 28, 28], dtype=np.float32)),
            axis=0)
        inputs = inputs * (2./255) - 1.
        inputs = np.reshape(inputs, [seq_len*2, 28*28])
        target_batch.append(targets)
        input_batch.append(inputs)
    return target_batch, input_batch


def local_mnist_input(mnist_gen, seq_len, batch_size, img_path, lbl_path):
    """Use locally downloaded images to feed the model.

    Create an MNIST image, label generator. Call this during the training
    loop to get current data.

    """
    try:
        final_o_data, final_i_data = _gen_sequence(
                mnist_gen, seq_len, batch_size)
    except:
        mnist_gen = disk_data(img_path, lbl_path)
        final_o_data, final_i_data = _gen_sequence(
            mnist_gen, seq_len, batch_size)
    return final_o_data, final_i_data


def builtin_mnist_input(mnist_dset, seq_len, batch_size):
    """Use the MNIST input module from the tutorials.

    Before, create an mnist dataset using the tutorial api. Call this during
    the training loop.

    """
    target_seq_batch = []
    input_seq_batch = []
    for _ in range(seq_len):
        img_batch, lbl_batch = mnist_dset.train.next_batch(batch_size)
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
                 img_path=".",
                 lbl_path=".",
                 builtin_input=True,
                 builtin_download_path="/tmp/tensorflow/mnist/input_data",
                 tb_dir="tb/dnc"):
    """Run training loop."""
    if builtin_input:
        print("loading mnist data: ")
        mnist = input_data.read_data_sets(builtin_download_path, one_hot=True)
        print("Done")
    else:
        mnist = disk_data(img_path, lbl_path)
    graph = tf.Graph()

    with graph.as_default():
        with tf.Session() as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            i_data = tf.placeholder(tf.float32, [batch_size, seq_len*2, 28*28])
            o_data = tf.placeholder(tf.float32, [batch_size, seq_len*2, seq_width])

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
            apply_gradients = RMS_train(loss)
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(
                tb_dir, graph=tf.get_default_graph())

            for epoch in range(iterations+1):
                if builtin_input:
                    curr_o_data, curr_i_data = builtin_mnist_input(
                        mnist, seq_len, batch_size)
                else:
                    curr_o_data, curr_i_data = local_mnist_input(
                        mnist, seq_len, batch_size, img_path, lbl_path)
                feed_dict = {i_data: curr_i_data, o_data: curr_o_data}
                current_loss, predictions, _, _ = sess.run(
                    [loss, output, apply_gradients, update_if_stateful],
                    feed_dict=feed_dict)
                if epoch % 100 == 0:
                    print("Epoch {}: Loss {}".format(epoch, current_loss))

    print("Final inputs:")
    print(curr_i_data)
    print("Final targets:")
    print(curr_o_data)
    print("Final predictions:")
    print(predictions)


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
                 img_path=os.path.join(FLAGS.data_dir, FLAGS.train_imgs),
                 lbl_path=os.path.join(FLAGS.data_dir, FLAGS.train_lbls),
                 builtin_input=FLAGS.builtin_input,
                 builtin_download_path=FLAGS.data_dir,
                 tb_dir=os.path.join(FLAGS.tb_dir, FLAGS.tb_train))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-e", "--epochs", type=int, default=25000,
        help="Number of epochs (minus one) model trains.")
    parser.add_argument(
        "-tb", "--tb_dir", type=str,
        default="tb/dnc",
        help="Path for folder containing TensorBoard data.")
    parser.add_argument(
        "--tb_train", type=str, default="train",
        help="TensorBoard extension for training data.")
    parser.add_argument(
        "-d", "--data_dir", type=str,
        default="/tmp/tensorflow/mnist/input_data",
        help="-bi false: path to image, label folder. -bi true: path to save downloaded inputs")
    parser.add_argument(
        "--train_imgs", type=str,
        default="mnist_train_images",
        help="Train image file within data_dir, used if -bi true.")
    parser.add_argument(
        "--train_lbls", type=str,
        default="mnist_train_labels",
        help="Train label file within data_dir, used if -bi true")
    parser.add_argument(
        "-s", "--seq_len", type=int, default=3,
        help="Length of (nonzero) input sequence.")
    parser.add_argument(
        "-b", "--batch_size", type=int, default=25,
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
        "-bi", "--builtin_input", action='store_false', default=True,
        help="TensorFlow tutorial mnist input.")
    parser.add_argument(
        "-W", "--bit_len", type=int, default=1024,
        help="Length of a slot in memory.")
    parser.add_argument(
        "-N", "--mem_len", type=int, default=6,
        help="Slots in memory.")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
