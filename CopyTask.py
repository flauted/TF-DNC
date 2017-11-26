#!/usr/bin/env python
"""dncdbg.py."""
import numpy as np
import tensorflow as tf
import os
from tensorflow.python import debug as tf_debug
from DNCv2 import DNC


class MLPModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

    def __call__(self, inputs):
        """Control the DNC."""
        with tf.variable_scope("L1"):
            W1 = tf.get_variable(
                "layer1_weights",
                shape=[self.input_size, self.hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b1 = tf.get_variable(
                "layer1_bias",
                shape=[32],
                initializer=tf.zeros_initializer())
            l1_evidence = tf.matmul(inputs, W1) + b1
            l1_act = tf.nn.tanh(l1_evidence)
            self.W1 = W1
            self.b1 = b1
        with tf.variable_scope("L2"):
            W2 = tf.get_variable(
                "layer2_weights",
                shape=[self.hidden_size, self.output_size],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b2 = tf.get_variable(
                "layer2_bias",
                shape=[self.output_size],
                initializer=tf.zeros_initializer())
            l2_evidence = tf.matmul(l1_act, W2) + b2
            l2_act = tf.nn.tanh(l2_evidence)
            # a^y = l2_act = net([x_t, r_{t-1}])
            self.W2 = W2
            self.b2 = b2
        return l2_act


def data(seq_len, seq_width):
    """Generate inputs for DNC sequence copy task."""
    con = np.random.randint(0, seq_width, size=seq_len)
    seq = np.zeros((seq_len, seq_width))
    seq[np.arange(seq_len), con] = 1
    zer = np.zeros((seq_len, seq_width))
    final_i_data = np.concatenate((seq, zer), axis=0)
    final_o_data = np.concatenate((zer, seq), axis=0)
    return final_i_data, final_o_data


def main(argv=None):
    """Run training loop."""
    seq_len = 6
    seq_width = 4  # seems to be bit_len
    iterations = 1000
    final_i_data, final_o_data = data(seq_len, seq_width)

    graph = tf.Graph()

    with graph.as_default():
        with tf.Session() as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            dnc = DNC(
                input_size=seq_width,
                output_size=seq_width,
                seq_len=seq_len,
                mem_len=10,
                bit_len=4,
                num_heads=2,
                softmax_allocation=False)
            dnc.install_controller(
                MLPModel(dnc.nn_input_size, 32, dnc.nn_output_size))
            output = dnc()
            with tf.name_scope("Eval"):
                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=output,
                        labels=dnc.o_data))
                with tf.name_scope("Regularizers"):
                    regularizers = (tf.nn.l2_loss(dnc.controller.W1) +
                                    tf.nn.l2_loss(dnc.controller.W2) +
                                    tf.nn.l2_loss(dnc.controller.b1) +
                                    tf.nn.l2_loss(dnc.controller.b2))
                loss += 5e-4 * regularizers
            with tf.name_scope("Train"):
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=0.001).minimize(loss)

            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(
                "tb/dnc", graph=tf.get_default_graph())

            for epoch in range(0, iterations+1):
                feed_dict = {dnc.i_data: final_i_data,
                             dnc.o_data: final_o_data}
                current_loss, _, predictions = sess.run(
                    [loss, optimizer, output],
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
