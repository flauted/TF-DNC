"""Define a differentiable neural computer with alternative allocation.

The differentiable neural computer was introduced by Graves, A., et. al. [2016]
as a neural network model with a dynamic memory modeled after the modern
CPU and RAM setup. """
import tensorflow as tf
import numpy as np


class DNC:
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
        input_size (arg)
        output_size (arg)
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
        mem (zeros, ``[batch_size, mem_len, bit_len]``): The internal
            memory matrix.
        usage_vec (zeros, ``[batch_size, mem_len]``): The usage vector.
        link_mat (zeros, ``[batch_size, mem_len, mem_len]``): The temporal
            link matrix.
        precedence_weight (zeros, ``[batch_size, mem_len]``): Part of write
            control.
        read_weights (+~zeros, ``[batch_size, mem_len, num_heads]``): Weight
            for reading memory.
        write_weights (rand, ``[batch_size, mem_len]``): Weight for
            writing memory.
        read_vecs (rand, ``[batch_size, bit_len, num_heads]``): Init
            read state.
        i_data (``[batch_size, seq_len*2, input_size]``): Placeholder for
            inputs.
        o_data (``[batch_size, seq_len*2, output_size]``): Placeholder for
            outputs.

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
        self.input_size = input_size
        self.output_size = output_size
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
        with tf.variable_scope("DNC/Mem_Man/zero_state"):
            self.mem = tf.zeros(
                [batch_size, mem_len, bit_len], name="memory")
            # u is N x 1
            self.usage_vec = tf.zeros([batch_size, mem_len], name="usage_vec")
            # L is N x N
            self.link_mat = tf.zeros(
                [batch_size, mem_len, mem_len],
                name="link_mat")
            # p is N x 1
            self.precedence_weight = tf.zeros(
                [batch_size, mem_len], name="precedence_weight")

            # HEAD VARIABLES
            self.read_weights = tf.fill(
                [batch_size, mem_len, num_heads], 1e-6, name="read_weights")
            self.write_weights = tf.fill(
                [batch_size, mem_len], 1e-6, name="write_weights")
        with tf.variable_scope("DNC/zero_state/read_vec"):
            self.read_vecs = tf.truncated_normal(
                    [batch_size, bit_len, num_heads],
                    stddev=0.1,
                    name="read_vec")

        # NETWORK VARIABLES
        self.i_data = tf.placeholder(
            tf.float32,
            [batch_size, seq_len*2, self.input_size],
            name="input")
        self.o_data = tf.placeholder(
            tf.float32,
            [batch_size, seq_len*2, self.output_size],
            name="output")

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

    def _controller(self, x, read_vecs, reuse):
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
        with tf.variable_scope("DNC/x_r_cat/"):
            # [x_t; r_{t-1}]
            reshape_read_vecs = tf.reshape(
                read_vecs, [self.batch_size, self.num_heads*self.bit_len])
            inputs = tf.concat([x, reshape_read_vecs], 1)

        with tf.variable_scope("controller/controller", reuse=reuse):
            l2_act = self.controller(inputs)

            with tf.variable_scope("prediction"):
                # v_t = W^y * a^y
                nn_out_weights = tf.get_variable(
                    "nn_out_weights",
                    shape=[self.nn_output_size, self.output_size],
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

    def interface_partition(self, interface_vec, return_alloc_strength=None):
        """Partition the interface vector into the memory controls.

        We use a strided slicing approach. This allows for eventual
        extensibility to batches, as well as nice graph visualization.
        After partitioning, the controls are possibly reshaped and ran
        through activation functions to preserve their domain. ::

                              VARIABLE REFERENCE
              Key             Math       Shape*      Domain
             ------------------------------------------------------
              read_keys       k^r_t[i]   B x R x W
              read_strengths  B^r_t[i]   B x 1 x R   [0, inf)
              write_key       k^w_t      B x 1 x W
              write_strength  B^w_t      B x 1       [0, inf)
              erase_vec       e_t        B x W       [0 1]
              write_vec       v_t        B x W
              free_gates      f_t[i]     B x 1 x R   [0 1]
              alloc_gate      g^a_t      B x 1       [0 1]
            [ alloc_strength  B^a_t      B x 1       [0, inf)     ]+
              write_gate      g^w_t      B x 1       [0 1]
              read_modes      pi_t[i]    B x 3 x 4   SOFTMAX SIMPLEX

        *B stands for ``batch_size``, R for ``num_heads``, and W
        for ``bit_len`` (consistent with paper). Index ``[i]`` corresponds to
        dimension with size R.

        +Only emitted when ``softmax_allocation`` is true.

        The variable reference table is provided to clarify the
        slicing operations. The reference helps with the obscure
        implementation, and so too does the TensorBoard graph.

        Args:
            interface_vec (``[batch_size, interface_size]``): The memory
                interface values.

        Returns:
            A dictionary with key-value pairs as described in
            the chart. Notice "Key" in the chart corresponds to
            keys for the return dict.

        """
        with tf.variable_scope("interface"):
            entries_per_part = [
                self.num_heads*self.bit_len,
                self.num_heads,
                self.bit_len,
                1,
                self.bit_len,
                self.bit_len,
                self.num_heads,
                1,
                1 if return_alloc_strength else 0,
                1,
                self.num_heads*3]
            start_idxs = np.cumsum([0] + entries_per_part)
            int_parts = {}
            with tf.variable_scope("read_keys"):
                int_parts["read_keys"] = tf.reshape(
                    interface_vec[:, start_idxs[0]:start_idxs[1]],
                    [self.batch_size, self.num_heads, self.bit_len])
            with tf.variable_scope("read_strength"):
                int_parts["read_strengths"] = tf.expand_dims(
                    1 + tf.nn.softplus(
                        interface_vec[:, start_idxs[1]:start_idxs[2]]),
                    1)
            with tf.variable_scope("write_key"):
                int_parts["write_key"] = tf.expand_dims(
                    interface_vec[:, start_idxs[2]:start_idxs[3]],
                    1)
            with tf.variable_scope("write_strength"):
                int_parts["write_strength"] = 1 + tf.nn.softplus(
                    interface_vec[:, start_idxs[3]:start_idxs[4]])
            with tf.variable_scope("erase_vec"):
                int_parts["erase_vec"] = tf.nn.sigmoid(
                    interface_vec[:, start_idxs[4]:start_idxs[5]])
            with tf.variable_scope("write_vec"):
                int_parts["write_vec"] = interface_vec[
                    :, start_idxs[5]:start_idxs[6]]
            with tf.variable_scope("free_gates"):
                int_parts["free_gates"] = tf.expand_dims(
                    tf.nn.sigmoid(
                        interface_vec[:, start_idxs[6]:start_idxs[7]]),
                    1)
            with tf.variable_scope("alloc_gate"):
                int_parts["alloc_gate"] = tf.nn.sigmoid(
                    interface_vec[:, start_idxs[7]:start_idxs[8]])
            if return_alloc_strength:
                with tf.variable_scope("alloc_strength"):
                    int_parts["alloc_strength"] = 1 + tf.nn.softplus(
                        interface_vec[:, start_idxs[8]:start_idxs[9]])
            with tf.variable_scope("write_gate"):
                int_parts["write_gate"] = tf.nn.sigmoid(
                    interface_vec[:, start_idxs[9]:start_idxs[10]])
            with tf.variable_scope("Read_modes"):
                int_parts["read_modes"] = tf.nn.softmax(
                    tf.reshape(
                        interface_vec[:, start_idxs[10]:start_idxs[11]],
                        [self.batch_size, 3, self.num_heads]))
        return int_parts

    @staticmethod
    def content_lookup(memory, key, strength):
        r"""Lookup from memory.

        A key vector - emitted by controller - is compared
        to content of each location in memory according to
        a similarity measurement. The sim scores determine a
        weighting that can be used by the read heads for
        recall or by the write heads to modify memory.

        Corresponds to

        .. math::

            D(u, v) &= \frac{u \cdot v}{\lVert u \rVert \lVert v \rVert},\\
            C(M, k, \beta)[i] &= \frac{exp(D(k,M[i,\cdot]) \beta)} {\sum_j(exp(D(k,M[j,\cdot]) \beta))}

        """
        with tf.variable_scope("CosineSimilarity"):
            norm_mem = tf.nn.l2_normalize(
                memory, 2, name="norm_mem")
            norm_key = tf.nn.l2_normalize(key, 1, name="norm_key")
            with tf.variable_scope("similarity"):
                similarity_z = tf.squeeze(
                    tf.matmul(
                        norm_mem, norm_key, transpose_b=True, name="lookup"))
        with tf.variable_scope("scaling"):
            similarity_scaled = tf.multiply(
                similarity_z, strength, name="str_scale")
            similarity_a = tf.nn.softmax(similarity_scaled, 1)
        return similarity_a

    @staticmethod
    def usage_update(
            prev_usage_vec, write_weights, read_weights, free_gates):
        r"""Update the usage vector.

        Comparing to the paper,

        .. math::

            u_t = (u_{t-1} + w^w_{t-1} - (u_{t-1} \odot w^w_{t-1})) \odot \psi_t

        such that

        .. math::

            \psi_t = \prod_{i=1}^R (1-f^i_t w^{r,i}_{t-1}).

        Notice that :math:`f^i_t` is the ith of ``num_heads`` free gates
        emitted by the controller. Each free gate is in ``[0,1]``. And,
        :math:`w^w_{t-1}` is the old computed write weight vector. Finally,
        :math:`w^{r,i}_{t-1}` is the old computed read weight.

        Args:
            prev_usage_vec: A real valued usage vector of shape
                ``batch_size x mem_len``.
            write_weights: A corner-vector (weaker all-positive unit vector)
                of shape ``batch_size x mem_len``.
            read_weights: A corner-vector (weaker all-positive unit vector)
                of shape ``batch_size x mem_len x 1`` for each head,
                making an effective shape of ``mem_len x num_heads``.
            free_gates: A vector of shape ``batch_size x 1 x num_heads``
                with each element in ``[0, 1]``.
        Returns:
            The new usage vector according to the above formulae.

        """
        # write_weights = tf.stop_gradient(write_weights)
        with tf.variable_scope("usage_after_write"):
            usage_after_write = prev_usage_vec + (
                1 - prev_usage_vec) * write_weights
        with tf.variable_scope("usage_after_read"):
            psi = tf.reduce_prod(
                1 - read_weights*free_gates, axis=2)
        new_usage_vec = usage_after_write * psi
        return new_usage_vec

    @staticmethod
    def softmax_allocation_weighting(usage_vec, alloc_strength):
        """Retrieve the writing allocation weight.

        The 'usage' is a number between 0 and 1. The `nonusage` is
        then computed by subtracting 1 from usage. Afterwards, we
        sharpen the `nonusage` to serve as the allocation weighting.

        As for interpretation, the ``1 x mem_len`` allocation weighting has
        a weight for each `bit` in the memory matrix. The value of the
        entry, in `[0, 1]`, controls how much the write head may alter
        the bit corresponding with that entry.

        The original paper proposed write allocation based on
        a tricky nondifferentiable sorting operation. This code implements
        allocation using a softmax operation instead, as proposed by
        Ben-Ari, I., Bekker, A. J., [2017] in "Differentiable Memory
        Allocation Mechanism For Neural Computing."

        In practice, the usage vector may become negative. This
        may be due to numerical error or may be a result of the softmax.

        Args:
            usage_vec: The ``batch_size x mem_len`` corner-vector.
            alloc_strength: A learned parameter from the interface of
                shape ``batch_size x 1`` in ``[0, 1]``.

        Returns:
            Calculated allocation weights.

        """
        nonusage = tf.subtract(1., usage_vec, name="nonusage")
        alloc_weights = tf.nn.softmax(nonusage*alloc_strength, dim=1)
        return alloc_weights

    def sorting_allocation_weighting(self, usage_vec):
        r"""Retrieve the writing allocation weight.

        The 'usage' is a number between 0 and 1. The `nonusage` is
        then computed by subtracting 1 from usage. Afterwards, we
        sharpen the `nonusage` to serve as the allocation weighting.

        As for interpretation, the ``1 x mem_len`` allocation weighting has
        a weight for each `bit` in the memory matrix. The value of the
        entry, in `[0, 1]`, controls how much the write head may alter
        the bit corresponding with that entry.

        First, we sort the indices, comprising

        .. math::

            \phi_t = \text{SortIndicesAscending(u_t)}

        In practice, we use TensorFlow's built-in sorting, giving us
        :math:`\phi_t` ``= freelist``. Since ``tf.nn.top_k`` sorts
        descendingly by default, we sort ``nonusage``, returning
        ``sorted_nonusage`` in addition to ``freelist``.

        Then, comparing to the paper we have

        .. math::

            a_t[\phi_t[j]] = (1 - u_t[phi_t[j]]) \prod_{i=1}^{j-1} u_t[phi_t[i]].

        Implementing this, notice that :math:`(1 - u_t[phi_t[j]])`
        ``= sorted_nonusage[j]``. Then see that
        :math:`\prod_{i=1}^{j-1} u_t[phi_t[i]]` can be computed for all
        `j` using an exclusive cumprod. (Note that it is assumed when `j=1,`
        the term is `1`.) Then, we calculate ``sorted_alloc``, meaning `in
        order`, by element-wise multiplying ``sorted_nonusage`` and our
        cumulative product vector. Finally, we revert the allocation
        weighting to the original ordering. We gather the ``freelist``
        entries of ``sorted_alloc``.

        Args:
            usage_vec: The ``batch_size x mem_len`` vector.

        Returns:
            Calculated allocation weights.

        """
        nonusage = tf.multiply(-1., usage_vec, name="nonusage")
        sorted_nonusage, freelist = tf.nn.top_k(
            nonusage, k=self.mem_len, name="sort")
        sorted_usage = tf.multiply(-1., sorted_nonusage, name="sorted_usage")
        prod_sorted_use = tf.cumprod(
            sorted_usage, axis=1, exclusive=True, name="prod_sorted_use")
        sorted_alloc = tf.multiply(
            (1 - sorted_usage), prod_sorted_use, name="sorted_allocation")
        unsorted_alloc_list = [
            tf.gather(sorted_alloc[b, :], freelist[b, :], name="unsort")
            for b in range(self.batch_size)]
        alloc_weights = tf.stack(
            unsorted_alloc_list, axis=0, name="collect_batches")
        return alloc_weights

    @staticmethod
    def update_write_weights(
            alloc_weights, write_content_lookup, alloc_gate, write_gate):
        """Update write weights to reflect allocation decisions.

        Comparing to the formula, we have

        .. math::

            g^w_t[g^a_t a_t + (1 - g^a_t)c^w_t]

        where :math:`g^a_t` in ``[0, 1]`` is the allocation gate, :math:`a_t`
        is the allocation corner-vector, :math:`g^w_t` in ``[0, 1]`` is the
        write gate, :math:`(1 - g^a_t)` is the "unallocation gate," and
        :math:`c^w_t` is the writing content lookup.

        Args:
            alloc_weights: The tensor of size ``batch_size x mem_len``.
            write_content_lookup: A unit vector of size
                ``batch_size x mem_len``.
            write_gate: A scalar in ``[0, 1]`` for each batch entry having
                shape ``batch_size x 1``.
            alloc_gate: A scalar in ``[0, 1]`` for each batch entry having
                shape ``batch_size x 1``.

        Returns:
            The new write weights, a corner-vector of size
                ``batch_size x mem_len``.

        """
        scaled_alloc = tf.multiply(
            alloc_gate,
            alloc_weights,
            name="alloc_g_by_alloc")
        unalloc_gate = tf.subtract(
            1., alloc_gate, name="g_unalloc")
        lookup_alloc = tf.multiply(
            unalloc_gate,
            write_content_lookup,
            name="unalloc_g_by_cw")
        write_locations = tf.add(
            scaled_alloc, lookup_alloc, name="write_locations")
        write_weights = tf.multiply(
            write_gate,
            write_locations,
            name="write_g_by_loc")
        return write_weights

    @staticmethod
    def erase_and_write_memory(
            old_memory, write_weights, erase_vec, write_vec):
        r"""Erase and write the memory matrix.

        Comparing to the paper, we have

        .. math::

            M_t = M_{t-1} \odot ( [[1]] - w^w_t (e_t)^T) + w^w_t (v_t)^T

        where :math:`w^w_t` is the computed write vector, :math:`e_t` is the
        emitted erase vector, and :math:`v_t` is the emitted write vector.
        Also, :math:`[[1]]` denotes a matrix of ones.

        As for implementation, we sidestep the transposition by expanding
        :math:`e_t, v_t` to ``batch_size x 1 x bit_len`` and expanding
        :math:`w^w_t` as ``batch_size x mem_len x 1``.

        Args:
            old_memory: A matrix of size ``mem_len x bit_len``.
            write_weights: The computed write weighting corner-vector of size
                ``mem_len x 1``.
            erase_vec: The emitted erase vector of size
                ``batch_size x bit_len``.
            write_vec: The emitted write vector of size
                ``batch_size x bit_len``.
        Returns:
            The updated memory matrix.

        """
        with tf.variable_scope("ERASE"):
            write_weights = tf.expand_dims(write_weights, -1)
            with tf.variable_scope("erase_matrix"):
                erase_vec = tf.expand_dims(erase_vec, 1)
                erase_matrix = 1 - tf.matmul(write_weights, erase_vec)
            with tf.variable_scope("erase_memory"):
                erased_mem = old_memory*erase_matrix

        with tf.variable_scope("WRITE"):
            with tf.variable_scope("write_matrix"):
                write_vec = tf.expand_dims(write_vec, 1)
                add_matrix = tf.matmul(write_weights, write_vec)
            with tf.variable_scope("write_memory"):
                new_memory = erased_mem + add_matrix
        return new_memory

    def update_temporal_link(
            self, prev_link_mat, write_weights, prev_precedence):
        r"""Update the temporal link matrix.

        Comparing to the paper, we have

        .. math::

            L_t[i,j] &= (1-w^w_t[i]-w^w_t[j]) L_{t-1}[i,j] + w^w_t[i] p_{t-1}[j] \\
            L_t[i,i] &= 0

        where :math:`w^w_t` is the write weight corner-vector and :math:`p_t`
        is the precedence corner-vector.

        The actual implementation is different. Instead we broadcast
        write_weights into a ``(batch_size) x mem_len x mem_len``
        matrix ``expanded_weights`` of the form ::

            [[ w^w[1]    w^w[1]    ...  w^w[1]    ]
             [ w^w[2]    w^w[2]    ...  w^w[2]    ]
             ...
             [ w^w[m.l.] w^w[m.l.] ...  w^w[m.l.] ]],

        then performing ``1 - expanded_weights - transpose(expanded_weights)``.
        Then we element-wise multiply the previous temporal link matrix.
        At this point we have :math:`(1 - w^w_t[i]-w^w_t[j]) L_{t-1}[i,j]`
        for all :math:`i,j.`

        Then, we multiply the write weights by the precedence weights and
        add to the previous operations. This comprises
        :math:`... + w^w_t[i] p_{t-1}[j]` for all :math:`i,j.`.

        Finally, we subtract our result from an identity matrix to
        eliminate self-links in the temporal link matrix.

        Args:
            prev_link_mat: The old ``batch_size x mem_len x mem_len``
                temporal link matrix.
            write_weights: The ``batch_size x mem_len`` write weighting
                corner-vector.
            prev_precedence: The ``batch_size x mem_len`` precedence
                corner-vector.

        Returns:
            The new temporal link matrix.

        """
        with tf.variable_scope("link_mat"):
            # L_t[i,j] for all i,j : i != j
            write_weights = tf.expand_dims(
                write_weights, 2, name="write_weights")
            prev_precedence = tf.expand_dims(
                prev_precedence, 1, name="prev_precedence")
            expanded_weights = tf.matmul(
                write_weights,
                tf.ones([self.batch_size, 1, self.mem_len]),
                name="expanded_write_weights")
            first_part = tf.multiply(
                (1. - expanded_weights - tf.transpose(
                    expanded_weights, [0, 2, 1])),
                prev_link_mat,
                name="first_part")
            second_part = tf.matmul(
                write_weights, prev_precedence,
                name="second_part")
            # Lt[i,i] = 0 for all i
            I = tf.eye(self.mem_len, dtype=tf.float32, name="IdtyMat")
            un_I = tf.subtract(1., I, name="unEye")
            new_link_mat = un_I * (first_part + second_part)
        return new_link_mat

    @staticmethod
    def update_precedence(prev_precedence, write_weights):
        r"""Update the precedence weight vector.

        Comparing to the paper, we have

        .. math::

            p_t = [ 1 - \sum_{i} (w^w_t[i]) ] p_{t-1} + w^w_t,

        which is implemented exactly as written.

        Args:
            prev_precedence: The old ``batch_size x mem_len`` corner-vector.
            write_weights: The current ``batch_size x mem_len`` corner-vector.

        Returns:
            The updated precedence weighting.

        """
        reset_factor = 1 - tf.reduce_sum(write_weights, 1, keep_dims=True)
        new_precedence = reset_factor * prev_precedence + write_weights
        return new_precedence

    @staticmethod
    def update_read_weights(prev_read_weights,
                            link_mat,
                            read_content_lookup,
                            read_modes):
        r"""Update the read weights.

        Comparing to the paper, we have

        .. math::

            w^{r,i}_t = \pi^i_t[1]b^i_t + \pi^i_t[2]c^{r,i}_t + \pi^i_t[3]f^1_t

        where :math:`w^{r,i}_t` is the read weight for read head :math:`i`,
        :math:`pi^i_t` is the read mode vector for read head :math:`i,` and

        .. math::

            f^1_t &= L_t w^{r,i}_{t-1}, \\
            b^i_t &= (L_t)^T w^{r,i}_{t-1} \text{ and } \\
            c^{r,i}_t &= C(M_t, k^{r,i}_t, \beta^{r,i}_t). \\

        Args:
            prev_read_weights: The old ``batch_size x mem_len`` corner-vector
                for each read head, making an effective shape of ``batch_size x
                mem_len x num_heads``.
            link_mat: The current ``batch_size x mem_len x mem_len`` temporal
                link matrix.
            read_content_lookup: The ``batch_size x mem_len x 1`` corner-vector
                for each read head, making an effective shape of
                ``batch_size x mem_len x num_heads``.
            read_modes: The ``batch_size x 3 x 1`` unit vector for each
                read head, making an effective shape of
                ``batch_size x 3 x num_heads``.

        Returns:
            The new read weights.

        """
        with tf.variable_scope("forw_weights"):
            # mem_len x num_heads
            forw_w = tf.expand_dims(read_modes[:, 2, :], 1)*tf.matmul(
                link_mat, prev_read_weights)

        with tf.variable_scope("cont_weights"):
            # mem_len x num_heads
            cont_w = tf.expand_dims(read_modes[:, 1, :], 1)*read_content_lookup

        with tf.variable_scope("back_weights"):
            # mem_len x num_heads
            back_w = tf.expand_dims(read_modes[:, 0, :], 1)*tf.matmul(
                link_mat, prev_read_weights, transpose_a=True)

        with tf.variable_scope("read_weights"):
            # mem_len x num_heads
            read_weights = back_w + cont_w + forw_w
        return read_weights

    @staticmethod
    def read_memory(memory_matrix, read_weights):
        """Read off memory.

        Args:
            memory_matrix: The ``batch_size x mem_len x bit_len`` memory
                matrix.
            read_weights: The ``batch_size x mem_len`` corner-vector for
                each read head, making an effective shape of 
                ``num_heads x mem_len``.

        Returns:
            The read (past tense) real-valued vectors of size 
                ``batch_size x 1 x bit_len`` for each read head, making 
                an effective shape of ``batch_size x num_heads x bit_len``.

        """
        with tf.variable_scope("READ"):
            read_vecs = tf.transpose(
                    tf.matmul(
                        memory_matrix,
                        read_weights,
                        transpose_a=True,
                        name="weighted_read"),
                    [0, 2, 1]),
        return read_vecs

    def _interact_with_memory(self, nn_out, interface_vec, reuse):
        """Step up m.

        Receive input data and a set of read vectors from
        memory matrix at the previous timestep. Emit output data
        and interface vector defining memory interactions at
        current timestep.
        """
        with tf.variable_scope("Mem_Man/Mem_Man/mo", reuse=reuse):
            int_parts = self.interface_partition(
                interface_vec, return_alloc_strength=self.softmax_allocation)
            with tf.variable_scope("write_allocation"):
                with tf.variable_scope("usage"):
                    self.usage_vec = self.usage_update(
                        self.usage_vec,
                        self.write_weights,
                        self.read_weights,
                        int_parts["free_gates"])

                with tf.variable_scope("alloc_weights"):
                    if self.softmax_allocation:
                        alloc_weights = self.softmax_allocation_weighting(
                            self.usage_vec, int_parts["alloc_strength"])
                    else:
                        alloc_weights = self.sorting_allocation_weighting(
                            self.usage_vec)

                with tf.variable_scope("writing_lookup_weights"):
                    write_content_lookup = self.content_lookup(
                        self.mem,
                        int_parts["write_key"],
                        int_parts["write_strength"])

                with tf.variable_scope("write_weights"):
                    self.write_weights = self.update_write_weights(
                        alloc_weights,
                        write_content_lookup,
                        int_parts["alloc_gate"],
                        int_parts["write_gate"])

            self.mem = self.erase_and_write_memory(
                self.mem,
                self.write_weights,
                int_parts["erase_vec"],
                int_parts["write_vec"])

            with tf.variable_scope("read_allocation"):
                self.link_mat = self.update_temporal_link(
                    self.link_mat, self.write_weights, self.precedence_weight)

                with tf.variable_scope("precidence_weight"):
                    self.precedence_weight = self.update_precedence(
                        self.precedence_weight, self.write_weights)

                with tf.variable_scope("reading_lookup_weights"):
                    read_content_lookup = self.content_lookup(
                        self.mem,
                        int_parts["read_keys"],
                        int_parts["read_strengths"])

                with tf.variable_scope("read_weights"):
                    self.read_weights = self.update_read_weights(
                        self.read_weights,
                        self.link_mat,
                        read_content_lookup,
                        int_parts["read_modes"])

            self.read_vecs = self.read_memory(
                self.mem,
                self.read_weights)

        return self.read_vecs

    def timestep_out(self, nn_out, read_vecs, reuse):
        """Construct the output y_t from the nn_out and read memory."""
        with tf.variable_scope("y_t", reuse=reuse):
            read_vecs_out_weight = tf.get_variable(
                "readout_weights",
                shape=[self.num_heads*self.bit_len, self.output_size],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        with tf.variable_scope("DNC/y_t/"):
            reshape_read_vecs = tf.reshape(
                read_vecs,
                [self.batch_size, self.num_heads*self.bit_len],
                name="Reshape_r_t")
            access_state = tf.matmul(
                reshape_read_vecs, read_vecs_out_weight, name="access_state")
            y = tf.add(nn_out, access_state, name="y_t")
        return y

    def __call__(self):
        """Run the dnc on a sequence."""
        seq_prediction = []
        with tf.variable_scope("DNC"):
            with tf.variable_scope("x_t"):
                seq_list = tf.unstack(self.i_data, axis=1)
            for t, seq in enumerate(seq_list):
                nn_out, int_vec = self._controller(
                    seq, self.read_vecs, reuse=t > 0)
                read_vecs = self._interact_with_memory(
                    nn_out, int_vec, reuse=t > 0)
                y_t = self.timestep_out(nn_out, read_vecs, reuse=t > 0)
                seq_prediction.append(y_t)
            with tf.variable_scope("prediction/"):
                concat_output = tf.stack(
                    seq_prediction, axis=1, name="collect_out")
                squeeze_output = tf.squeeze(concat_output)
        return squeeze_output

