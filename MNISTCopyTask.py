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
from tensorflow.python import debug as tf_debug
from DNCv2 import DNC


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
            l2_evidence = l2_conv + b2
            l2_act = tf.nn.relu(l2_evidence)
            # a^y = l2_act = net([x_t, r_{t-1}])
            self.K2 = K2
            self.b2 = b2
        with tf.variable_scope("FC1"):
            conv_out = tf.reshape(l2_act, [self.batch_size, 64*4*4])
            fc1_input = tf.concat([conv_out, read_vecs], axis=1)
            W = tf.get_variable(
                "fc1_weights",
                shape=[64*4*4+self.read_vec_size, self.output_size],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(
                "fc1_bias",
                shape=[self.output_size],
                initializer=tf.zeros_initializer())
            fc1_out = tf.matmul(fc1_input, W) + b
            fc1_act = tf.nn.relu(fc1_out)
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


def gen_sequence(label_img_gen, seq_len, batch_size):
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


def mnist_input(mnist_generator, seq_len, batch_size, img_path, lbl_path):
    try:
        final_o_data, final_i_data = gen_sequence(
                mnist_generator, seq_len, batch_size)
    except:
        my_gen = disk_data(img_path, lbl_path)
        final_o_data, final_i_data = gen_sequence(my_gen, seq_len, batch_size)
    return final_o_data, final_i_data


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


def run_training(seq_len=3,
                 seq_width=10,
                 iterations=25000,
                 mem_len=12,
                 bit_len=1024,
                 num_read_heads=4,
                 batch_size=50,
                 softmax_alloc=True,
                 img_path="/media/dylan/DATA/mnist/mnist_train_images",
                 lbl_path="/media/dylan/DATA/mnist/mnist_train_labels",
                 tb_dir="tb/dnc"):
    """Run training loop."""
    my_gen = disk_data(img_path, lbl_path)
    graph = tf.Graph()

    with graph.as_default():
        with tf.Session() as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            dnc = DNC(
                input_size=28*28,
                output_size=seq_width,
                seq_len=seq_len,
                mem_len=mem_len,
                bit_len=bit_len,
                batch_size=batch_size,
                num_heads=num_read_heads,
                softmax_allocation=softmax_alloc)
            dnc.install_controller(
                ConvModel(
                    dnc.nn_input_size,
                    dnc.nn_output_size,
                    bit_len*num_read_heads,
                    batch_size))
            output = dnc()
            loss = evaluate(seq_len=seq_len, seq_width=seq_width,
                            labels=dnc.o_data, logits=output)
            apply_gradients = update(loss)
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(
                tb_dir, graph=tf.get_default_graph())

            for epoch in range(iterations+1):
                o_data, i_data = mnist_input(my_gen, seq_len, batch_size,
                                             img_path, lbl_path)
                feed_dict = {dnc.i_data: i_data, dnc.o_data: o_data}
                current_loss, predictions, _ = sess.run(
                    [loss, output, apply_gradients],
                    feed_dict=feed_dict)
                if epoch % 100 == 0:
                    print("Epoch {}: Loss {}".format(epoch, current_loss))
            print("Final inputs:")
            print(i_data)
            print("Final targets:")
            print(o_data)
            print("Final predictions:")
            print(predictions)


def main(argv=None):
    run_training()


if __name__ == '__main__':
    tf.app.run()
