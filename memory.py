"""Create a memory module for the DNC."""
import tensorflow as tf
import numpy as np
import collections


AccessState = collections.namedtuple(
    'AccessState', ('mem',
                    'usage',
                    'link',
                    'precedence',
                    'read_weights',
                    'write_weights',
                    'read_vecs'))

class Memory:
    r"""Implement the memory module of the differentiable neural computer.

    The interaction is as follows

    * Split up controller interface vector.

    * Make write weights

        - Free gates determine "whether the most recently read locations
          can be freed."
        - Retention vector :math:`\psi` "represents by how much each
          location will NOT be freed by the free gates." Each entry is
          in ``[0, 1]``.
        - Usage: A location has high usage if it has been retained by
          the free gates (:math:`\psi` near 1), AND usage of the location
          at the last timestep was high or the location was just written
          to (previous write weights near 1).

    A nice proof that usage stays in ``[0, 1]`` is provided in section 6.3
    of "Implementation and Optimization of Differentiable Neural Computers"
    by C. Hsin.

        - Allocation: A sharpened, inverted usage whose elements are in
          ``[0, 1]`` and  sum to at most 1. ``Allocate``, or write to
          locations with a low usage.
        - Write weights: The controller gate determines whether to write
          to a new location (alloc gate), a location with high content
          similarity (1 - alloc gate), or not write at all (write gate).
          Each entry is in ``[0, 1]`` and the vector sums to at most
          1.

    Thinking of the weight weights as a probability distribution,
    the remainder of ``1 - sum(weights)`` is the probability of
    accessing no memory location at all (called null operations).

    * Write memory. Alter memory location ``[i, j]`` as follows

        - Scale down by 1 minus the ith entry of write weights times
          jth entry of erase vector, a factor in ``[0, 1]``.
        - Add ith entry of write weights times jth entry of write vector.
        - In this sense, the write weights determine a scale for each
          row, while the write and erase vectors carry information for
          all the columns.

    * Make read weights

        - The temporal link matrix retains the writing order. An entry
          ``[h, i, j]`` encodes "the degree to which location `i` ... is
          written to after location `j`" by head `h`. Each row and column 
          of L defines a probability distribution over the locations, 
          with nulls.
        - Precedence: Each element represents how close that location
          was to the most recent writing, per head. If precedence is high,
          the memory location has changed a lot recently.
        - The following happens for each read head
        - Forward/Backward weighting: Redistribute the previous read
          weighting based on the TLM. The backward weighting is determined
          by transposing the TLM, effectively reversing the ordering.
        - Read weights: The read modes is a probability distribution
          (elements in ``[0, 1]`` summing to 1 exactly) that controls how
          the read weights function. If element 1 is high, the reading
          happens in the reverse order of writing (backward weighting
          dominates). If element 2 is high, the reading happens on
          locations where there is a high degree of similarity between
          the location and the read key. If element 3 is high, the reading
          happens in the order of the TLM.

    By "order," obviously we do not mean literal iteration; the action
    does not occur in sequence. It means that e.g. if element 1 of the
    read modes is high, the information in the oldest memory location is
    factored in highest in the head's read vector.)

    * Read memory. Multiply the memory by the read weights for each head
      to give the read (past tense) vectors.

    """

    def __init__(self,  # NOQA
                 mem_len,
                 bit_len,
                 n_read_heads,
                 n_write_heads,
                 batch_size,
                 softmax_allocation):
        self.mem_len = mem_len
        self.bit_len = bit_len
        self.n_read_heads = n_read_heads
        self.n_write_heads = n_write_heads
        self.batch_size = batch_size
        self.softmax_allocation = softmax_allocation

    def interface_partition(self, interface_vec, return_alloc_strength=None):
        """Partition the interface vector into the memory controls.

        After partitioning, the controls are possibly reshaped and ran
        through activation functions to preserve their domain. ::

                              VARIABLE REFERENCE
              Key             Math       Shape*      Domain
             ------------------------------------------------------
              read_keys**     k^r_t      B x W x R   R
              read_strengths  B^r_t      B x 1 x R   [0, inf)
              write_key**     k^w,h_t    B x W x H   R
              write_strength  B^w,h_t    B x 1 x H   [0, inf)
              erase_vec       e_t        B x H x W   [0 1]
              write_vec       v_t        B x H x W   R
              free_gates      f^r_t      B x 1 x R   [0 1]
              alloc_gate      g^a,h_t    B x 1 x H   [0 1]
             [alloc_strength+ B^a_t      B x 1       [0, inf) ]
              write_gate      g^w,h_t    B x 1 x H   [0 1]
              read_modes      pi^r_t                 SIMPLEX in R dim
                forward     pi^r_t[:H]   B x H x R   
                backward    pi^r_t[H:2H] B x H x R
                content     pi^r_t[H]    B x 1 x R

        *B stands for ``batch_size``, R for ``n_read_heads``, and W
        for ``bit_len`` (consistent with paper). We use H for 
        ``n_write_heads``. Index ``[i]`` corresponds to dimension with size 
        R, index ``[j]`` corresponds to dimension with size H.

        +Only emitted when ``softmax_allocation`` is true.

        **In the paper, keys are shaped W x _.

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
                self.n_read_heads*self.bit_len,
                self.n_read_heads,
                self.n_write_heads*self.bit_len,
                self.n_write_heads,
                self.bit_len*self.n_write_heads,
                self.bit_len*self.n_write_heads,
                self.n_read_heads,
                self.n_write_heads,
                self.n_write_heads if return_alloc_strength else 0,
                self.n_write_heads,
                self.n_read_heads * (self.n_write_heads*2 + 1)]
            start_idxs = np.cumsum([0] + entries_per_part)

            intrfc = {}

            with tf.variable_scope("read_keys"):
                keys_r = interface_vec[:, start_idxs[0]:start_idxs[1]]
                intrfc["read_keys"] = tf.reshape(
                    keys_r, [self.batch_size, self.bit_len, self.n_read_heads])

            with tf.variable_scope("read_strength"):
                B_r_hat = interface_vec[:, start_idxs[1]:start_idxs[2]]
                B_r = 1 + tf.nn.softplus(B_r_hat)
                intrfc["read_strengths"] = tf.expand_dims(B_r, 1)

            with tf.variable_scope("write_key"):
                keys_w = interface_vec[:, start_idxs[2]:start_idxs[3]]
                intrfc["write_key"] = tf.reshape(
                    keys_w, 
                    [self.batch_size, self.bit_len, self.n_write_heads])

            with tf.variable_scope("write_strength"):
                B_w_hat = interface_vec[:, start_idxs[3]:start_idxs[4]]
                B_w = 1 + tf.nn.softplus(B_w_hat)
                intrfc["write_strength"] = tf.expand_dims(B_w, 1)

            with tf.variable_scope("erase_vec"):
                e_hat = interface_vec[:, start_idxs[4]:start_idxs[5]]
                e_ = tf.nn.sigmoid(e_hat)
                intrfc["erase_vec"] = tf.reshape(
                    e_, [self.batch_size, self.n_write_heads, self.bit_len])

            with tf.variable_scope("write_vec"):
                v_ = interface_vec[:, start_idxs[5]:start_idxs[6]]
                intrfc["write_vec"] = tf.reshape(
                    v_, [self.batch_size, self.n_write_heads, self.bit_len])

            with tf.variable_scope("free_gates"):
                f_r_hat = interface_vec[:, start_idxs[6]:start_idxs[7]]
                f_r = tf.nn.sigmoid(f_r_hat)
                intrfc["free_gates"] = tf.expand_dims(f_r, 1)

            with tf.variable_scope("alloc_gate"):
                g_a_hat = interface_vec[:, start_idxs[7]:start_idxs[8]]
                g_a = tf.nn.sigmoid(g_a_hat)
                intrfc["alloc_gate"] = tf.expand_dims(g_a, 1)

            if return_alloc_strength:
                with tf.variable_scope("alloc_strength"):
                    B_a_hat = interface_vec[:, start_idxs[8]:start_idxs[9]]
                    B_a = 1 + tf.nn.softplus(B_a_hat)
                    intrfc["alloc_strength"] = tf.expand_dims(B_a, 1)

            with tf.variable_scope("write_gate"):
                g_w_hat = interface_vec[:, start_idxs[9]:start_idxs[10]]
                g_w = tf.nn.sigmoid(g_w_hat)
                intrfc["write_gate"] = tf.expand_dims(g_w, 1)

            with tf.variable_scope("Read_modes"):
                pi_hat = interface_vec[:, start_idxs[10]:start_idxs[11]]
                pi_r_hat = tf.reshape(
                    pi_hat, [self.batch_size, self.n_write_heads*2+1, self.n_read_heads])
                pi_r = tf.nn.softmax(pi_r_hat)
                intrfc["read_modes"] = pi_r

        return intrfc

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

        Args:
            memory: The ``batch_size x mem_len x bit_len`` memory matrix.
            key: The ``batch_size x num_heads x mem_len`` key vector.
            strength: The ``batch_size x 1 x num_heads`` strength vector.

        Returns:
            The cosine similarity between the key and the memory of shape
            ``batch_size x mem_len x num_keys``.

        """
        with tf.variable_scope("CosineSimilarity"):
            norm_mem = tf.nn.l2_normalize(
                memory, 2, name="norm_mem")
            norm_key = tf.nn.l2_normalize(key, 1, name="norm_key")
            with tf.variable_scope("similarity"):
                similarity_z = tf.matmul(
                        norm_mem, norm_key, name="lookup")

        with tf.variable_scope("scaling"):
            similarity_scaled = tf.multiply(
                similarity_z, strength, name="str_scale")
            similarity_a = tf.nn.softmax(similarity_scaled, 1)
        return similarity_a

    @staticmethod
    def usage_update(prev_usage, write_weights, read_weights, free_gates):
        r"""Update the usage vector.

        Comparing to the paper,

        .. math::

            u_t = (u_{t-1} + w^w_{t-1} - (u_{t-1} \odot w^w_{t-1})) \odot \psi_t

        such that

        .. math::

            \psi_t = \prod_{i=1}^R (1-f^i_t w^{r,i}_{t-1}).

        Notice that :math:`f^i_t` is the ith of ``n_read_heads`` free gates
        emitted by the controller. Each free gate is in ``[0,1]``. And,
        :math:`w^w_{t-1}` is the old computed write weight vector. Finally,
        :math:`w^{r,i}_{t-1}` is the old computed read weight.

        Args:
            prev_usage_vec: A real valued usage vector of shape
                ``batch_size x mem_len``.
            write_weights: A corner-vector (weaker all-positive unit vector)
                of shape ``batch_size x mem_len x n_write_heads``.
            read_weights: A tensor of corner vectors for each read head having
                shape ``batch_size x mem_len x n_read_heads``.
            free_gates: A vector of shape ``batch_size x 1 x n_read_heads``
                with each element in ``[0, 1]``.
        Returns:
            The new usage vector according to the above formulae.

        """
        # write_weights = tf.stop_gradient(write_weights)

        # small write weight -> big weight'
        # big weight' for all column -> big effect'
        # big effect' -> small effect
        with tf.variable_scope("effect"):
            effect = 1 - tf.reduce_prod(1 - write_weights, [2])
        with tf.variable_scope("usage_after_write"):
            usage_after_write = prev_usage + (1 - prev_usage) * effect
        with tf.variable_scope("usage_after_read"):
            psi = tf.reduce_prod(1 - read_weights*free_gates, axis=2)
        new_usage_vec = tf.multiply(usage_after_write, psi, name="new_usage")
        return new_usage_vec

    def softmax_allocation_weighting(self, usage_vec, head_alloc_strength):
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
                shape ``batch_size x 1 x n_write_heads`` in ``[0, 1]``.

        Returns:
            Calculated allocation weights.

        """
        nonusage = tf.subtract(1., usage_vec, name="nonusage")
        alloc_weights = tf.nn.softmax(nonusage*head_alloc_strength, dim=1)
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
            Calculated allocation weights of shape ``batch_size x mem_len``.

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

    def alloc_update(
            self, usage, write_gates, alloc_strength=None, softmax=None):
        """Allocate memory locations for each write head.

        After every allocation, update the dummy usage vector.

        Args:
            usage: A tensor of shape ``batch_size x mem_len``.
            write_gates: A tensor of shape ``batch_size x 1 x n_write_heads``.

        Returns:
            A tensor of shape ``batch_size x mem_len x n_write_heads``.

        """
        alloc_list = []
        if softmax and alloc_strength is None:
            raise Exception("You must pass alloc strength if using softmax")
        for head in range(self.n_write_heads):
            with tf.variable_scope("head_alloc_weight"):
                if not softmax:
                    alloc_list.append(self.sorting_allocation_weighting(usage))
                else:
                    alloc_list.append(self.softmax_allocation_weighting(
                                           usage, alloc_strength[:, :, head]))
            if head is not self.n_write_heads:
                with tf.variable_scope("update_usage_dummy"):
                    head_write_g = write_gates[:, :, head]
                    usage += ((1 - usage) * head_write_g * alloc_list[head])
        full_alloc_weights = tf.stack(alloc_list, axis=-1, name="alloc_weight")
        return full_alloc_weights

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
            alloc_weights: The tensor of size 
                ``batch_size x mem_len x n_write_heads``.
            write_content_lookup: A unit vector of size
                ``batch_size x mem_len x n_write_heads``.  
            write_gate: A tensor with elements in ``[0, 1]`` having shape
                ``batch_size x 1 x n_write_heads``.
            alloc_gate: A tensor with elements in ``[0, 1]`` having shape 
                ``batch_size x 1 x n_write_heads``.

        Returns:
            The new write weights, a corner-vector (in last dim) of size
                ``batch_size x mem_len x n_write_heads``.

        """
        scaled_alloc = alloc_gate * alloc_weights
        unalloc_gate = 1. - alloc_gate
        lookup_alloc = unalloc_gate * write_content_lookup
        write_locations = scaled_alloc + lookup_alloc
        write_weights = write_gate * write_locations
        return write_weights

    @staticmethod
    def erase_and_write_memory(old_mem, write_weights, erase_vec, write_vec):
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
            old_memory: A matrix of size ``batch_size x mem_len x bit_len``.
            write_weights: The computed write weighting corner-vector of size
                ``batch_size x mem_len x n_write_heads``.
            erase_vec: The emitted erase vector of size
                ``batch_size x n_write_heads x bit_len``.
            write_vec: The emitted write vector of size
                ``batch_size x n_write_heads x bit_len``.
        Returns:
            The updated memory matrix.

        """
        with tf.variable_scope("Erase"):
            with tf.variable_scope("erase_matrix"):
                erase_matrix = 1 - tf.matmul(write_weights, erase_vec)
            erased_mem = tf.multiply(old_mem, erase_matrix, name="erase_mem")

        with tf.variable_scope("Write"):
            with tf.variable_scope("write_matrix"):
                add_matrix = tf.matmul(write_weights, write_vec)
            new_mem = tf.add(erased_mem, add_matrix, name="write_mem")
        return new_mem

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
            prev_link_mat: The old 
                ``batch_size x n_write_heads x mem_len x mem_len`` temporal 
                link matrix.
            write_weights: The ``batch_size x mem_len x n_write_heads`` 
                write weighting corner-vector.
            prev_precedence: The ``batch_size x mem_len x n_write_heads`` 
                precedence corner-vector.

        Returns:
            The new temporal link matrix.

        """
        with tf.variable_scope("link_mat"):
            # L_t[i,j] for all i,j : i != j
            write_weights = tf.transpose(write_weights, [0, 2, 1])
            prev_precedence = tf.transpose(prev_precedence, [0, 2, 1])
            write_weights_i = tf.expand_dims(
                write_weights, 2, name="write_weights_i")
            write_weights_j = tf.expand_dims(
                write_weights, 3, name="write_weights_j")
            prev_precedence = tf.expand_dims(
                prev_precedence, 3, name="prev_precedence_j")
            with tf.variable_scope("weights_factor"):
                weights_factor = 1 - write_weights_i - write_weights_j
            with tf.variable_scope("fresh_links"):
                new_links = prev_precedence * write_weights_i
            _new_link_mat = weights_factor * prev_link_mat + new_links
            # Lt[i,i] = 0 for all i
            new_link_mat = tf.matrix_set_diag(
                _new_link_mat, 
                tf.zeros([self.batch_size, self.n_write_heads, self.mem_len]),
                name="zero_diag")
        return new_link_mat

    @staticmethod
    def update_precedence(prev_precedence, write_weights):
        r"""Update the precedence weight vector.

        Comparing to the paper, we have

        .. math::

            p_t = [ 1 - \sum_{i} (w^w_t[i]) ] p_{t-1} + w^w_t,

        which is implemented exactly as written.

        Args:
            prev_precedence: The old ``batch_size x mem_len x n_write_heads`` 
                corner-vector.
            write_weights: The current ``batch_size x mem_len x n_write_heads``
                corner-vector.

        Returns:
            The updated precedence weighting.

        """
        reset_factor = 1 - tf.reduce_sum(write_weights, 1, keep_dims=True)
        new_precedence = reset_factor * prev_precedence + write_weights
        return new_precedence

    def update_read_weights(self,
                            prev_read_weights,
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
            prev_read_weights: The old tensor of corner-vectors for each read
                head, with shape of ``batch_size x mem_len x n_read_heads``.
            link_mat: The current 
                ``batch_size x n_write_heads x mem_len x mem_len`` temporal
                link matrix.
            read_content_lookup: A tensor of corner-vectors for each read head,
                with shape ``batch_size x mem_len x n_read_heads``.
            read_modes: A tensor of unit vectors for each read head, having
                shape ``batch_size x 2*n_write_heads + 1 x n_read_heads``.

        Returns:
            The new read weights.

        """
        with tf.variable_scope("expanded_prev_read_weights"):
            expanded_read_weights = tf.stack(
                [prev_read_weights] * self.n_write_heads, 1)

        with tf.variable_scope("forw_weights"):
            forw_w = tf.matmul(link_mat, expanded_read_weights)
        cont_w = read_content_lookup
        with tf.variable_scope("back_weights"):
            back_w = tf.matmul(link_mat, expanded_read_weights,
                               transpose_a=True)

        with tf.variable_scope("read_weights"):
            # Take into account the action's prob in the read modes distrib.
            with tf.variable_scope("forw_part"):
                forw_prob = read_modes[:, :self.n_write_heads, :]
                forw_prob = tf.expand_dims(forw_prob, 2)
                # AND consider multiple write heads
                forw_pt = tf.reduce_sum(forw_w * forw_prob, axis=1)
            with tf.variable_scope("back_part"):
                back_prob = read_modes[
                    :, self.n_write_heads:2*self.n_write_heads, :]
                back_prob = tf.expand_dims(back_prob, 2)
                back_pt = tf.reduce_sum(back_w * back_prob, axis=1)
            with tf.variable_scope("cont_part"):
                cont_prob = read_modes[:, 2*self.n_write_heads, :]
                cont_prob = tf.expand_dims(cont_prob, 1)
                cont_pt = cont_w * cont_prob
            read_weights = back_pt + cont_pt + forw_pt
        return read_weights

    @staticmethod
    def read_memory(memory, read_weights):
        """Read off memory.

        Args:
            memory_matrix: The ``batch_size x mem_len x bit_len`` memory
                matrix.
            read_weights: A tensor of corner-vectors for each read head,
                with shape ``n_read_heads x mem_len``.

        Returns:
            The read (past tense) real-valued vectors for each read head; a
            tensor of shape ``batch_size x n_read_heads x bit_len``.

        """
        with tf.variable_scope("Read"):
            read_vecs = tf.matmul(read_weights, memory, transpose_a=True)
        return read_vecs

    def _interact_with_memory(self, interface_vec, prev_state):
        """Step up m.

        Receive input data and a set of read vectors from
        memory matrix at the previous timestep. Emit output data
        and interface vector defining memory interactions at
        current timestep.

        Args:
            interface_vec: The ``batch_size x 1 x interface_size`` interface
                vector emitted by the controller.
            prev_state: The AccessState named tuple of state values memory,
                usage, link matrix, precedence, write weights, read weights,
                and previous read vectors.

        Returns:
            The read vectors, a tensor of shape
                ``batch_size x n_read_heads x bit_len`` of values read
                from memory.
            An AccessState named tuple with the new state values.

        """
        with tf.variable_scope("Mem_Man"):
            intrfc = self.interface_partition(
                interface_vec, return_alloc_strength=self.softmax_allocation)
            with tf.variable_scope("write_allocation"):
                with tf.variable_scope("usage"):
                    usage_vec = self.usage_update(
                        prev_state.usage,
                        prev_state.write_weights,
                        prev_state.read_weights,
                        intrfc["free_gates"])

                with tf.variable_scope("alloc_weights"):
                    alloc_weights = self.alloc_update(
                        usage_vec,
                        intrfc["write_gate"],
                        softmax=self.softmax_allocation,
                        alloc_strength=(intrfc["alloc_strength"] 
                                        if self.softmax_allocation else None))

                with tf.variable_scope("writing_lookup_weights"):
                    write_content_lookup = self.content_lookup(
                        prev_state.mem,
                        intrfc["write_key"],
                        intrfc["write_strength"])

                with tf.variable_scope("write_weights"):
                    write_weights = self.update_write_weights(
                        alloc_weights,
                        write_content_lookup,
                        intrfc["alloc_gate"],
                        intrfc["write_gate"])

            mem = self.erase_and_write_memory(
                prev_state.mem,
                write_weights,
                intrfc["erase_vec"],
                intrfc["write_vec"])

            with tf.variable_scope("read_allocation"):
                link_mat = self.update_temporal_link(
                    prev_state.link, write_weights, prev_state.precedence)

                with tf.variable_scope("precedence_weight"):
                    precedence_weight = self.update_precedence(
                        prev_state.precedence, write_weights)

                with tf.variable_scope("reading_lookup_weights"):
                    read_content_lookup = self.content_lookup(
                        prev_state.mem,
                        intrfc["read_keys"],
                        intrfc["read_strengths"])

                with tf.variable_scope("read_weights"):
                    read_weights = self.update_read_weights(
                        prev_state.read_weights,
                        link_mat,
                        read_content_lookup,
                        intrfc["read_modes"])

            read_vecs = self.read_memory(
                mem,
                read_weights)

        return read_vecs, AccessState(mem=mem,
                                      usage=usage_vec,
                                      write_weights=write_weights,
                                      link=link_mat,
                                      precedence=precedence_weight,
                                      read_weights=read_weights,
                                      read_vecs=read_vecs)
