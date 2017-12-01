r"""Define a differentiable neural computer with alternative allocation.

The differentiable neural computer was introduced by Graves, A., et. al. [2016]
as a neural network model with a dynamic memory modeled after the modern
CPU and RAM setup.
"""
import tensorflow as tf
import collections
from memory import Memory


AccessState = collections.namedtuple(
    'AccessState', ('mem',
                    'usage',
                    'link',
                    'precedence',
                    'read_weights',
                    'write_weights',
                    'read_vecs'))


class DNC(tf.nn.rnn_cell.RNNCell):
    """Create a differentiable neural computer.

    The DNC is a sequence-to-sequence model that is completely
    differentiable. It features a built-in memory.

    Comparing to the paper glossary available at
    https://www.readcube.com/articles/supplement?doi=10.1038%2Fnature20101&index=12&ssl=1&st=acd80c7ede3649cb0f4345bcdc01ec12&preview=1
    we have ::

        W <=> bit_len (memory word size)
        N <=> mem_len (number of memory locations)
        R <=> num_heads (number of read heads)

    Args:
        input_size (int): Size of a row of input to ``run`` method.
        output_size (int): Expected size of output from ``run``.
        seq_len (int): Number of rows of input.

    Keyword Args:
        mem_len (int, 256): Number of slots in memory.
        bit_len (int, 64): Length of a slot in memory.
        num_heads (int, 4): Number of read heads.
        batch_size (int, 1): Length of the batch.
        softmax_allocation (bool, True): Use the alternative softmax writing
            allocation or the original formulation.

    Attributes:
        output_width (arg, output_size)
        mem_len (arg)
        bit_len (arg)
        num_heads (arg)
        batch_size (arg)
        softmax_allocation (arg)
        interface_size (``num_heads*bit_len + 3*bit_len + 5*num_heads + 3``):
            Size of emitted interface vector.
        nn_input_size (``num_heads*bit_len + input_size``): Size of concatted
            read and input vector.
        nn_output_size (``output_size + interface_size``): Size of concatted
            prediction and interface vector.
        controller (``None``): A user defined callable (function / instance
            with a ``__call__`` method).
              NOTE: If you need ``nn_input_size`` or ``nn_output_size`` to
              define the controller, use `myDNC.install_controller(callable)`
              after initializing myDNC.

    """

    def __init__(self,  # NOQA
                 input_size,
                 output_size,
                 seq_len,
                 controller=None,
                 mem_len=256,
                 bit_len=64,
                 num_heads=4,
                 batch_size=1,
                 softmax_allocation=True):
        self.output_width = output_size
        self.mem_len = mem_len
        self.bit_len = bit_len
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.softmax_allocation = softmax_allocation
        # size of output from controller for memory interactions
        self.interface_size = num_heads*bit_len + 3*bit_len + 5*num_heads + 3
        if softmax_allocation:
            self.interface_size += 1
        # actual sizes after concat
        self.nn_input_size = num_heads * bit_len + input_size
        self.nn_output_size = output_size + self.interface_size
        self.controller = controller
        self._memory = Memory(
            mem_len, bit_len, num_heads, batch_size, softmax_allocation)

        self._access_shapes = AccessState(
            mem=tf.TensorShape([mem_len, bit_len]),
            usage=tf.TensorShape([mem_len]),
            link=tf.TensorShape([mem_len, mem_len]),
            precedence=tf.TensorShape([mem_len]),
            # HEAD VARIABLES
            read_weights=tf.TensorShape([mem_len, num_heads]),
            write_weights=tf.TensorShape([mem_len]),
            read_vecs=tf.TensorShape([num_heads, bit_len]))

    def zero_state(self):
        with tf.variable_scope("zero_state"):
            return AccessState(
                mem=tf.Variable(
                    tf.zeros([self.batch_size, self.mem_len, self.bit_len]),
                    trainable=False, name="mem"),
                usage=tf.Variable(
                    tf.zeros([self.batch_size, self.mem_len]),
                    trainable=False, name="usage"),
                link=tf.Variable(
                    tf.zeros([self.batch_size, self.mem_len, self.mem_len]),
                    trainable=False, name="link"),
                precedence=tf.Variable(
                    tf.zeros([self.batch_size, self.mem_len]),
                    trainable=False, name="prec"),
                read_weights=tf.Variable(
                    tf.fill([self.batch_size, self.mem_len, self.num_heads],
                        1e-6, name="read_weights"),
                    trainable=False, name="w_r"),
                write_weights=tf.Variable(
                    tf.fill([self.batch_size, self.mem_len],
                        1e-6, name="write_weights"),
                    trainable=False, name="w_w"),
                read_vecs=tf.Variable(
                    tf.truncated_normal(
                        [self.batch_size, self.num_heads, self.bit_len],
                        stddev=0.1, name="read_vec"),
                    trainable=False, name="read_vecs"))

    @property
    def output_size(self):
        return tf.TensorShape([self.output_width])

    @property
    def state_size(self):
        return self._access_shapes

    def install_controller(self, controller):
        r"""Determine the controller for the DNC.

        The input is expected to be a callable that maps from size
        ``1 x nn_input_size`` to size ``1 x nn_output_size.`` Recall
        that ``nn_input_size = input_size + num_heads*bit_len`` and
        that ``nn_output_size = output_size + interface_size.`` Note that
        the controller object is `not` responsible for emitting seperate
        prediction and interface vectors.

        The controller may be a function installed without ``()`` or
        an object with a ``__call__`` method. If the controller is an object,
        it may be initialized with DNC object attributes, especially
        ``myDNC.nn_input_size`` and ``myDNC.nn_output_size``.

        As for mathematical discussion, the controller maps the time
        step input :math:`x_t` concatenated with the interpreted
        information read from memory by each head, :math:`r^i_{t-1}`,
        to the prediction and the interface vector. The interface vector
        controls the memory interactions. Precisely

        .. math::

            \text{ctrlr}([x_t;r^1_{t-1};...;r^R_{t-1}]) \mapsto ([\hat{y}_t; \hat{\zeta}_t])

        where :math:`r^i_{t-1}` is the interpreted information from read vector
        `i` at the previous time step.

        To be clear, :math:`[\cdot ; \cdot]` denotes concatenation
        and :math:`[\hat{y}_t; \hat{\zeta}_t]` is the vector denoting
        the prediction evidence and the interface evidence. Outside
        the controller object, the DNC multiplies the vector emmited by
        the controller by the output weights :math:`W^y_t` and then
        again by the interface weights :math:`W^\zeta_t`. In other words,
        the DNC converts the ``1 x nn_output_size =
        1 x output_size + interface_size`` vector to one prediction of length
        ``output_size`` and another interface vector of length
        ``interface_size``.

        Args:
            controller: A callable to predict outputs and select
                interface variables. The controller must take only one
                argument, a tensor of size ``1 x nn_input_size``,
                and return only one tensor of size ``1 x nn_output_size.``

        Examples:
            We may use an initialized object::

                myDNC = DNC(...)
                controller = MLPClass(in_size=myDNC.nn_input_size,
                                      out_size=myDNC.nn_output_size,
                                      hidden_size=32)
                myDNC.install_controller(controller)

            In this case, ``controller`` `must` have a __call__ method
            taking only one argument: the input to the DNC at that timestep
            concatenated with the ``num_heads`` read vectors.

            Or, we may use a function::

                def net(x):
                    # x is [1, nn_input_size]
                    ...
                    # y is [1, nn_output_size]
                    return y
                myDNC.install_controller(net)

        """
        self.controller = controller

    def _controller(self, x, read_vecs):
        """Perform controller operations.

        Use the DNC's installed controller to make an inference
        given a sample from a time sequence ``x`` (formally ``x_t``)
        and the weighted vectors read from memory.

        Args:
            x ([batch_size, input_size]): The seq sample at the current
                timestep.
            read_vecs: Vectors interpreted from memory. Must be
                reshapable to [batch_size, num_heads*bit_len]. (From the paper,
                `read_vecs` would be [num_heads, bit_len].)
        Returns:
            The predicted mapping for `x_t', a tensor of shape
                [batch_size, output_size].
            The memory interface values, a tensor of shape
                [batch_size, interface_size].

        """
        with tf.variable_scope("x_r_cat"):
            # [x_t; r_{t-1}]
            reshape_read_vecs = tf.reshape(
                read_vecs, [self.batch_size, self.num_heads*self.bit_len])
            inputs = tf.concat([x, reshape_read_vecs], 1)

        l2_act = self.controller(inputs)

        with tf.variable_scope("prediction"):
            # v_t = W^y * a^y
            nn_out_weights = tf.get_variable(
                "nn_out_weights",
                shape=[self.nn_output_size, self.output_width],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            nn_out = tf.matmul(l2_act, nn_out_weights)

        with tf.variable_scope("interface"):
            # Zeta_t = W^Z * a^y
            interface_weights = tf.get_variable(
                "interface_weights",
                shape=[self.nn_output_size, self.interface_size],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            interface_vec = tf.matmul(l2_act, interface_weights)
        return nn_out, interface_vec

    def final_prediction(self, nn_out, read_vecs):
        """Construct the output y_t from the nn_out and read memory."""
        with tf.variable_scope("y_t"):
            read_vecs_out_weight = tf.get_variable(
                "readout_weights",
                shape=[self.num_heads*self.bit_len, self.output_width],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            reshape_read_vecs = tf.reshape(
                read_vecs,
                [self.batch_size, self.num_heads*self.bit_len],
                name="Reshape_r_t")
            interpreted_read_vecs = tf.matmul(
                reshape_read_vecs, read_vecs_out_weight, name="weighted_read")
            y = tf.add(nn_out, interpreted_read_vecs, name="y_t")
        return y

    def __call__(self, inputs, prev_state):
        """Run the dnc on a sequence."""
        with tf.variable_scope("DNC"):
            with tf.variable_scope("Controller"):
                nn_out, int_vec = self._controller(
                    inputs, prev_state.read_vecs)
            read_vecs, _state = self._memory._interact_with_memory(
                nn_out, int_vec, prev_state)
            y_t = self.final_prediction(nn_out, read_vecs)
        return y_t, _state
