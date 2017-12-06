#!/usr/bin/env python
"""Basic sequence copy task for DNC."""
import numpy as np
import tensorflow as tf
import os
from tensorflow.python import debug as tf_debug
from DNCv3 import DNC


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


def evaluate(seq_len=None, seq_width=None, labels=None, logits=None):
    with tf.name_scope("Eval"):
        mask = tf.concat(
            [tf.zeros([seq_len, seq_width]), tf.ones([seq_len, seq_width])],
            axis=0, name="mask")
        xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)
        masked_xent = mask * xent
        loss = tf.reduce_mean(masked_xent)
    return loss


def update(loss, learning_rate=1e-4, momentum=0.9, clip_low=-10, clip_high=10):
    with tf.name_scope("Train"):
        optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
        grads = optimizer.compute_gradients(loss)
        for i, (grad, var) in enumerate(grads):
            if grad is not None:
                grads[i] = (tf.clip_by_value(grad, clip_low, clip_high), var)
        update_op = optimizer.apply_gradients(grads)
    return update_op


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

            if stateful:
                update_ops = []
                for init_var, new_var in zip(initial_state, new_state):
                    update_ops.extend([init_var.assign(new_var)])

            loss = evaluate(seq_len=seq_len, seq_width=seq_width,
                            labels=o_data, logits=output)
            apply_gradients = update(loss)
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(
                "tb/dnc", graph=tf.get_default_graph())

            for epoch in range(0, iterations+1):
                curr_i_data, curr_o_data = data(seq_len, seq_width, batch_size)
                feed_dict = {i_data: curr_i_data, o_data: curr_o_data}

                predictions, current_loss, _, _ = sess.run(
                    [output, loss, apply_gradients,
                     update_ops if stateful else tf.no_op()],
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
