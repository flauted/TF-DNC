#!/usr/bin/env python
"""Basic sequence copy task for DNC."""
import numpy as np
import tensorflow as tf
import sys
import os
import argparse
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
            tf.summary.scalar("masked_xent", loss)
            apply_gradients = RMS_train(loss)
            summary_op = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
            tb_writer = tf.summary.FileWriter(
                tb_dir, graph=tf.get_default_graph())

            for epoch in range(0, iterations+1):
                curr_i_data, curr_o_data = data(seq_len, seq_width, batch_size)
                feed_dict = {i_data: curr_i_data, o_data: curr_o_data}

                predictions, current_loss, _, _, summary = sess.run(
                    [output,
                     loss,
                     apply_gradients,
                     update_if_stateful,
                     summary_op],
                    feed_dict=feed_dict)
                tb_writer.add_summary(summary, epoch)

                if epoch % 100 == 0:
                    print("Epoch {}: Loss {}".format(epoch, current_loss))

        tb_writer.close()

    print("Final inputs:")
    print(curr_i_data)
    print("Final targets:")
    print(curr_o_data)
    print("Final predictions:")
    print(predictions)


def main(_):
    run_training(seq_width=FLAGS.seq_width,
                 iterations=FLAGS.epochs,
                 mem_len=FLAGS.mem_len,
                 bit_len=FLAGS.bit_len,
                 num_read_heads=FLAGS.num_read_heads,
                 num_write_heads=FLAGS.num_write_heads,
                 batch_size=FLAGS.batch_size,
                 softmax_alloc=FLAGS.softmax,
                 stateful=FLAGS.stateful,
                 tb_dir=os.path.join(FLAGS.tb_dir, FLAGS.tb_ext))


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
        "--tb_ext", type=str, default="CopyTask",
        help="TensorBoard extension for training summary.")
    parser.add_argument(
        "-s", "--seq_len", type=int, default=6,
        help="Length of (nonzero) input sequence.")
    parser.add_argument(
        "-w", "--seq_width", type=int, default=4,
        help="Number of slots in the binary sequence.")
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
        "-sm", "--softmax", action='store_true', default=True,
        help="Use alternative softmax allocation.")
    parser.add_argument(
        "-W", "--bit_len", type=int, default=4,
        help="Length of a slot in memory.")
    parser.add_argument(
        "-N", "--mem_len", type=int, default=6,
        help="Slots in memory.")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
