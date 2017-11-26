#!/usr/bin/env python
"""dncdbg.py."""
import numpy as np
import tensorflow as tf
import os
import struct
from tensorflow.python import debug as tf_debug
from DNCv2 import DNC


class ConvModel:
    def __init__(self, input_size, output_size, read_vec_size):
        self.input_size = input_size
        self.output_size = output_size
        self.read_vec_size = read_vec_size

    def __call__(self, inputs):
        """Control the DNC."""
        input_imgs = tf.reshape(inputs[:, :28*28], [1, 1, 28, 28])
        read_vecs = tf.transpose(inputs[:, 28*28:])
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
            l2_evidence = l2_conv + b2
            l2_act = tf.nn.relu(l2_evidence)
            # a^y = l2_act = net([x_t, r_{t-1}])
            self.K2 = K2
            self.b2 = b2
        with tf.variable_scope("FC1"):
            conv_out = tf.reshape(l2_act, [64*4*4, 1])
            fc1_input = tf.concat([conv_out, read_vecs], axis=0)
            W = tf.get_variable(
                "fc1_weights",
                shape=[self.output_size, 64*4*4+self.read_vec_size],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(
                "fc1_bias",
                shape=[self.output_size, 1],
                initializer=tf.zeros_initializer())
            fc1_out = tf.matmul(W, fc1_input) + b
            fc1_act = tf.nn.tanh(fc1_out)
            output = tf.transpose(fc1_act)
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

    get_img = lambda idx: (label[idx], image[idx])
    for i in range(len(label)):
        yield get_img(i)


def gen_sequence(label_img_gen, seq_len):
    img_seq = []
    labels = np.zeros([seq_len, 10]) 
    for timestep in range(seq_len):
        label, img = next(label_img_gen)
        labels[timestep, label] = 1
        img_seq.append(img)
    targets = np.concatenate((np.zeros([seq_len, 10]), labels), axis=0)
    inputs = np.concatenate(
        (np.asarray(img_seq), 
        np.zeros([seq_len, 28, 28], dtype=np.float32)), axis=0)
    inputs = inputs * (2./255) - 1.
    inputs = np.reshape(inputs, [seq_len*2, 28*28])
    return targets, inputs


def main(argv=None):
    """Run training loop."""
    seq_len = 6
    seq_width = 10  # seems to be bit_len
    iterations = 9000
    my_gen = disk_data("/media/dylan/DATA/mnist/mnist_train_images", 
                       "/media/dylan/DATA/mnist/mnist_train_labels")
    graph = tf.Graph()

    with graph.as_default():
        with tf.Session() as sess:
            dnc = DNC(
                input_size=28*28,
                output_size=seq_width,
                seq_len=seq_len,
                mem_len=128,
                bit_len=8,
                num_heads=2)
            dnc.install_controller(
                ConvModel(dnc.nn_input_size, dnc.nn_output_size, 8*2))
            output = dnc()
            with tf.name_scope("Eval"):
                loss = tf.reduce_mean(
                    tf.losses.softmax_cross_entropy(
                        dnc.o_data,
                        output))
                with tf.name_scope("Regularizers"):
                    regularizers = (tf.nn.l2_loss(dnc.controller.K1) +
                                    tf.nn.l2_loss(dnc.controller.K2) +
                                    tf.nn.l2_loss(dnc.controller.b1) +
                                    tf.nn.l2_loss(dnc.controller.b2))
                loss += 5e-4 * regularizers
            with tf.name_scope("Train"):
                trainable_variables = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(
                    tf.gradients(loss, trainable_variables), 20)
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=0.001)
                trainable_variables = tf.trainable_variables()
                global_step = tf.get_variable(
                    name="global_step",
                    shape=[],
                    dtype=tf.int64,
                    initializer=tf.zeros_initializer(),
                    trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                 tf.GraphKeys.GLOBAL_STEP])
                train_step = optimizer.apply_gradients(
                    zip(grads, trainable_variables),
                    global_step=global_step)
            checker = tf.add_check_numerics_ops()  
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(
                "tb/dnc", graph=tf.get_default_graph())

            for epoch in range(0, iterations+1):
                final_o_data, final_i_data = gen_sequence(my_gen, seq_len)
                feed_dict = {dnc.i_data: final_i_data,
                             dnc.o_data: final_o_data}
                _, current_loss, predictions, _ = sess.run(
                    [checker, loss, output, train_step],
                    feed_dict=feed_dict)
                if epoch % 100 == 0:
                    print("Epoch {}: Loss {}".format(epoch, current_loss))
            print("Final inputs:")
            print(final_i_data)
            print("Final targets:")
            print(final_o_data)
            print("Final predictions:")
            print(predictions)


if __name__ == '__main__':
    tf.app.run()
