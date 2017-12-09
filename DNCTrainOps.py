import tensorflow as tf

def masked_xent(seq_len=None, seq_width=None, labels=None, logits=None):
    with tf.name_scope("Eval"):
        mask = tf.concat(
            [tf.zeros([seq_len, seq_width]), tf.ones([seq_len, seq_width])],
            axis=0, name="mask")
        xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)
        masked_xent = mask * xent
        loss = tf.reduce_mean(masked_xent)
    return loss


def RMS_train(
        loss, learning_rate=1e-4, momentum=0.9, clip_low=-10, clip_high=10):
    with tf.name_scope("Train"):
        optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
        grads = optimizer.compute_gradients(loss)
        for i, (grad, var) in enumerate(grads):
            if grad is not None:
                grads[i] = (tf.clip_by_value(grad, clip_low, clip_high), var)
        train_op = optimizer.apply_gradients(grads)
    return train_op


def state_update(initial_state, new_state, stateful=True):
    """Return a list of state variable update ops.

    Sometimes we want to create a stateful RNN. To do this, we make each
    initial state variable a ``tf.Variable`` object and perform an
    ``assign`` op on each variable.

    Args:
        initial_state: An iterable of ``tf.Variable`` objects.
        new_state: An iterable of Tensors.

    Keyword Args:
        stateful (bool, True): If we want a command line option for
            stateful operation, we want a dummy operation when stateful is
            off. That is, return ``tf.no_op()`` when stateful is ``True``.

    Returns:
        If stateful is ``True``, return a list of update operations that
            may be used by ``sess``. Else, return ``tf.no_op()``.

    Example: ::

        def run_training(stateful=None):
            inputs = get_input(...)
            cell = tf.nn.rnn_cell.BasicLSTMCell(..., state_is_tuple=True)
            init_state = cell.zero_state(...)
            var_init_state = tuple([tf.Variable(var) for var in init_state])
            output, new_state = tf.nn.dynamic_rnn(
                cell, inputs, initial_state=var_init_state)
            update_if_stateful = state_update(
                var_init_state, new_state, stateful=stateful)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer)
                for iterations in range(1000):
                    guess, _ = sess.run([output, update_if_stateful])

    """
    with tf.name_scope("Update_State"):
        if stateful:
            update_if_stateful = []
            for init_var, new_var in zip(initial_state, new_state):
                update_if_stateful.extend([init_var.assign(new_var)])
        else:
            update_if_stateful = tf.no_op()
    return update_if_stateful

