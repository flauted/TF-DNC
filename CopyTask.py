#!/usr/bin/env python
"""Basic sequence copy task for DNC."""
import numpy as np
import tensorflow as tf
import os
from tensorflow.python import debug as tf_debug
from DNCv3 import DNC
from DNCTrainOps import masked_xent, state_update, RMS_train


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
                shape=[self.hidden_size],
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


def data(seq_len, seq_width, batch_size):
    """Generate inputs for DNC sequence copy task."""
    final_o_data = []
    final_i_data = []
    for _ in range(batch_size):
        con = np.random.randint(0, seq_width, size=seq_len)
        seq = np.zeros((seq_len, seq_width))
        seq[np.arange(seq_len), con] = 1
        zer = np.zeros((seq_len, seq_width))
        i_data = np.concatenate((seq, zer), axis=0)
        o_data = np.concatenate((zer, seq), axis=0)
        final_o_data.append(o_data)
        final_i_data.append(i_data)
    return final_i_data, final_o_data


def run_training(seq_len=6,
                 seq_width=4,
                 iterations=50000,
                 mem_len=15,
                 bit_len=10,
                 num_read_heads=2,
                 num_write_heads=3,
                 batch_size=7,
                 softmax_alloc=False,
                 stateful=False,
                 tb_dir="tb/dnc"):
    """Run training loop."""
    graph = tf.Graph()

    with graph.as_default():
        with tf.Session() as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            i_data = tf.placeholder(tf.float32,
                                    [batch_size, seq_len*2, seq_width])
            o_data = tf.placeholder(tf.float32,
                                    [batch_size, seq_len*2, seq_width])

            dnc = DNC(
                input_size=seq_width,
                output_size=seq_width,
                seq_len=seq_len,
                mem_len=mem_len,
                bit_len=bit_len,
                n_read_heads=num_read_heads,
                n_write_heads=num_write_heads,
                batch_size=batch_size,
                softmax_allocation=softmax_alloc)
            dnc.install_controller(
                MLPModel(dnc.nn_input_size, 128, dnc.nn_output_size))
            initial_state = dnc.zero_state()
            output, new_state = tf.nn.dynamic_rnn(
                dnc,
                i_data,
                initial_state=initial_state,
                scope="DNC",
                parallel_iterations=1)

            update_if_stateful = state_update(
                initial_state, new_state, stateful=stateful)
            loss = masked_xent(seq_len=seq_len, seq_width=seq_width,
                               labels=o_data, logits=output)
            apply_gradients = RMS_train(loss)
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(
                "tb/dnc", graph=tf.get_default_graph())

            for epoch in range(0, iterations+1):
                curr_i_data, curr_o_data = data(seq_len, seq_width, batch_size)
                feed_dict = {i_data: curr_i_data, o_data: curr_o_data}

                predictions, current_loss, _, _ = sess.run(
                    [output, loss, apply_gradients, update_if_stateful],
                    feed_dict=feed_dict)

                if epoch % 100 == 0:
                    print("Epoch {}: Loss {}".format(epoch, current_loss))

    print("Final inputs:")
    print(curr_i_data)
    print("Final targets:")
    print(curr_o_data)
    print("Final predictions:")
    print(predictions)


def main(argv=None):
    run_training()


if __name__ == '__main__':
    tf.app.run()
