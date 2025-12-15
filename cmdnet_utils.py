#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 02 15:28:68 2025

@author: beck
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as KB


class modulation():
    '''Modulation class
    '''
    # Class Attribute
    name = 'Modulation object'
    # Initializer / Instance Attributes

    def __init__(self, mod_name):
        # modulation type, e.g., BPSK, QPSK, QAM16, QAM64, ASK4, ASK8
        self.mod_name = mod_name
        self.m = 0                  # modulation vector
        self.M = 0                  # modulation order
        self.compl = 1              # complex?
        self.mod2vec()              # Initialize output variables
        # a-priori probabilities (default: equal prob.)
        self.alpha = 1 / self.M * np.ones((self.M))
    # Instance methods

    def mod2vec(self):
        '''Create symbol alphabet vector m for for complex/real modulation mod:
            mod: Modulation string
            m: Symbol alphabet vector
            compl: Complex (1) or real (0) modulation
        '''
        if self.mod_name == 'BPSK':
            self.m = np.array([1, -1])
            self.compl = 0
        elif self.mod_name == 'QPSK' or self.mod_name == 'QAM4':
            # with Gray coding
            self.m = np.array([1, -1])
            self.compl = 1
        elif self.mod_name == 'QAM16':
            # with Gray coding
            self.m = np.array([-3, -1, 3, 1])
            self.compl = 1
        elif self.mod_name == 'ASK4':
            # with Gray coding
            self.m = np.array([-3, -1, 3, 1])
            self.compl = 0
        elif self.mod_name == 'QAM64':
            self.m = np.array([-7, -5, -1, -3, 7, 5, 1, 3])
            self.compl = 1
        elif self.mod_name == 'ASK8':
            self.m = np.array([-7, -5, -1, -3, 7, 5, 1, 3])
            self.compl = 0
        # falsch, richtig: 16*16=256
        # elif self.mod_name == 'QAM256':
        #     print('Gray coding einfügen!!!')
        #     self.m = np.array([-9, -7, -5, -3, -1, 1, 3, 5, 7, 9])
        #     self.compl = 1
        # elif self.mod_name == 'ASK16':
        #     print('Gray coding einfügen!!!')
        #     self.m = np.array([-9, -7, -5, -3, -1, 1, 3, 5, 7, 9])
        #     self.compl = 0
        else:
            print('Modulation not available.')
        # normalization # a = 1 / np.mean(self.m ** 2)
        a = np.sqrt(3 / (self.m.shape[0] ** 2 - 1))
        self.m = self.m * a
        # * (self.compl + 1) # effective size of symbol alphabet
        self.M = self.m.shape[0]
        return self.m, self.compl, self.M


def batch_dot(a, b):
    '''Computes the
    matrix vector product: A*b
    vector matrix product: a*B
    matrix product: A*B
    for a batch of matrices and vectors along dimension 0
    Shape of tensors decides operation
    '''
    if len(a.shape) == 3 and len(b.shape) == 2:
        y = np.einsum('nij,nj->ni', a, b)  # A*b
    elif len(a.shape) == 2 and len(b.shape) == 3:
        y = np.einsum('nj,nji->ni', a, b)  # b*A
    elif len(a.shape) == 3 and len(b.shape) == 3:
        y = np.einsum('nij,njk->nik', a, b)  # A*B
    else:
        print('Error')
        y = 0
    return y


def np_softmax(arg, axis=-1):
    '''Accurate softmax implementation in numpy
    '''
    exp_arg = np.exp(arg - np.max(arg, axis=axis, keepdims=True))
    softmax = exp_arg / np.expand_dims(np.sum(exp_arg, axis=axis), axis=axis)
    return softmax


def np_CMD(data, mod, it, delta, taui, binopt=0):
    '''Inference based on gumbel-softmax distribution -> non-convex and no analytical solution -> Gradient descent solution
    --------------------------------------------------------
    INPUT
    data: list of [yt, Ht, sigma] of MIMO system
    mod.alpha: Prior probabilities
    mod.m: Modulation alphabet
    it: Number of iterations
    taui: Inverse of softmax temperature (size of iterations + 1)
    delta: Gradient step size (size of iterations)
    binopt: Select special binary case
    OUTPUT
    xt: estimated symbols
    ft: estimated prob of symbols, one-hot vectors
    '''
    def np_CMD_bin(data, mod, it, delta, tau):
        '''Binary case with modulation alphabet m = [-1, 1] / [1, -1]
        alpha: Prior probabilities of -1
        '''
        yt = data[0]
        Ht = data[1]
        sigmat = data[2][:, np.newaxis]  # scalar?
        m = mod.m
        if (m == np.array([-1, 1])).all():
            alpha = mod.alpha[:, 0]
        else:
            alpha = mod.alpha[:, 1]
        # Gumbel softmax problem solved with gradient descent
        # Starting point
        s0 = 0 * alpha  # a-priori
        s = np.repeat(s0[np.newaxis, :], Ht.shape[0], axis=0)
        HH = batch_dot(np.transpose(Ht, (0, 2, 1)), Ht)
        yH = batch_dot(yt, Ht)
        for iteration in range(0, it):
            # functional iteration
            xt = np.tanh((np.log(1 / alpha - 1) + s) / (2 * tau[iteration]))
            xHH = batch_dot(xt, HH)
            grad_L = 1 / (2 * tau[iteration]) * (1 - xt ** 2) * \
                (xHH - yH) + sigmat ** 2 * np.tanh(s / 2)
            s = s - delta[iteration] * grad_L
        # Final evaluation of transmitted symbols
        xt = np.tanh((np.log(1 / alpha - 1) + s) / (2 * tau[-1]))

        if (m == np.array([-1, 1])).all():
            ft = np.concatenate(
                [(1 - xt[:, :, np.newaxis]) / 2, (1 + xt[:, :, np.newaxis]) / 2], axis=-1)
        else:
            ft = np.concatenate(
                [(1 + xt[:, :, np.newaxis]) / 2, (1 - xt[:, :, np.newaxis]) / 2], axis=-1)
        return ft, xt

    def np_CMD_multiclass(data, mod, it, delta, tau):
        '''Multiclass
        '''
        yt = data[0]
        Ht = data[1]
        sigmat = data[2][:, np.newaxis, np.newaxis]  # scalar?
        alpha = mod.alpha
        m = mod.m[np.newaxis, np.newaxis, :]
        # Gumbel softmax problem solved with gradient descent
        # Starting point
        G0 = np.zeros((Ht.shape[-1], m.shape[-1]))  # A-priori
        G = np.repeat(G0[np.newaxis, :], Ht.shape[0], axis=0)
        HH = batch_dot(np.transpose(Ht, (0, 2, 1)), Ht)
        yH = batch_dot(yt, Ht)
        for iteration in range(0, it):
            # functional iteration
            arg = (np.log(alpha) + G) / tau[iteration]
            ft = np_softmax(arg, -1)
            xt = np.sum(ft * m, axis=-1)
            xHH = batch_dot(xt, HH)
            grad_x = 1 / tau[iteration] * (ft * m - ft * xt[:, :, np.newaxis])
            grad_L = grad_x * \
                (xHH - yH)[:, :, np.newaxis] + sigmat ** 2 * (1 - np.exp(-G))
            G = G - delta[iteration] * grad_L
        # Final evaluation of transmitted symbols
        arg = (np.log(alpha) + G) / tau[-1]
        ft = np_softmax(arg, -1)
        xt = np.sum(ft * m, axis=-1)
        return ft, xt

    if mod.M == 2 and binopt == 1:
        ft, xt = np_CMD_bin(data, mod, it, delta, 1 / taui)
    else:
        ft, xt = np_CMD_multiclass(data, mod, it, delta, 1 / taui)

    return ft, xt


def CMD_initpar(M, L, typ, k=0, min_val=0.1, tau_max_scale=1):
    '''Function calculating a good starting point for CMD / loading starting point
    M: Number of classes / Modulation order
    L: Number of layers / iterations
    'typ': linear/constant
    k: parallel branches of CMDNet -> repetition of parameters
    min_val: minimum value of tau and delta
    '''
    tau_max = 1 / (M - 1) * tau_max_scale  # for 4ASK: tau_max_scale = 2
    delta_max = 1
    tau_min = min_val  # * tau_max # default: 0.1
    delta_min = min_val  # * delta_max # default: 0.1
    if typ.casefold() == 'linear':
        # Linear decrease
        tau0 = tau_max - (tau_max - tau_min) / L * np.linspace(0, L, L + 1)
        delta0 = delta_max - (delta_max - delta_min) / L * np.linspace(0, L, L)
        taui0 = 1 / tau0
    elif typ.casefold() == 'const':
        # Constant
        tau0 = tau_max * np.ones(L + 1)
        delta0 = delta_max * np.ones(L)
        taui0 = 1 / tau0
    else:
        # Default: only linear decrease in taui
        tau0 = tau_max - (tau_max - tau_min) / L * np.linspace(0, L, L + 1)
        delta0 = delta_max * np.ones(L)
        taui0 = 1 / tau0

    if k != 0:
        tau0 = tau0[:, np.newaxis].repeat((k), axis=-1)
        delta0 = delta0[:, np.newaxis].repeat((k), axis=-1)
        taui0 = 1 / tau0

    return delta0, taui0


class filename_module():
    '''Class responsible for file names
    '''
    # Class Attribute
    name = 'File name creator'
    # Initializer / Instance Attributes

    def __init__(self, path, typename, algo, fn_ext, sim_set, code_set=0, tf=2):
        # Inputs
        # Path
        self.ospath = path
        # What kind of data is being saved?
        self.typename = typename
        self.filename = ''
        self.path = ''
        self.pathfile = ''
        # MIMO simulation settings/parameters
        self.mod = sim_set['Mod']
        self.Nr = sim_set['Nr']
        self.Nt = sim_set['Nt']
        self.L = sim_set['L']
        self.algoname = algo
        # Filename extension for uniqueness
        self.fn_ext = fn_ext
        self.code = code_set
        self.tf = tf
        # Initialize
        self.generate_pathfile_MIMO()
    # Instance methods

    def generate_filename_MIMO(self):
        '''Generates file name
        '''
        if self.code:
            self.filename = self.typename + self.algoname + '_' + self.mod + '_{}_{}_{}_'.format(
                self.Nt, self.Nr, self.L) + self.code['code'] + self.code['dec'] + self.code['arch'] + '_' + self.fn_ext
        else:
            self.filename = self.typename + self.algoname + '_' + self.mod + \
                '_{}_{}_{}'.format(self.Nt, self.Nr, self.L) + \
                '_' + self.fn_ext
        return self.filename

    def generate_path_MIMO(self):
        '''Generates path name
        '''
        if self.tf == 2:
            self.path = os.path.join(self.ospath, self.mod, '{}x{}'.format(
                self.Nt, self.Nr), self.fn_ext)
        else:
            self.path = os.path.join(self.ospath, self.mod, '{}x{}'.format(
                self.Nt, self.Nr))
        return self.path

    def generate_pathfile_MIMO(self):
        '''Generates full path and filename
        '''
        self.generate_path_MIMO()
        self.generate_filename_MIMO()
        self.pathfile = os.path.join(self.path, self.filename)
        return self.pathfile


def print_time(time):
    '''Print time
    INPUT
    time: time in s
    OUTPUT
    time_str: time string
    '''
    m, s = divmod(time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    time_label = "{}:{:02d}:{:02d}:{:02d}".format(
        -int(d), int(h), int(m), int(np.round(s)))
    return time_label


def encoder(b, G):
    '''Encodes bitvector b with code given in generator matrix G of coding theory dimensions
    b: bit stream or bit vectors
    c: code bits
    '''
    if len(b.shape) == 1:
        b2 = b[:len(b) - np.mod(len(b), G.shape[0])]
        b2 = np.reshape(b2, (-1, G.shape[0]))
    else:
        # b2 = b.copy()
        b2 = b
    c = np.mod(np.dot(b2, G), 2).astype('int')
    return c


@tf.function  # (jit_compile = True)
def encoder_tf(b, G):
    """
    Encode bitvector(s) b with generator matrix G (coding-theory style).
    - b: 1D tensor (bit stream) or 2D tensor (batch of bit vectors). Values 0/1 expected.
    - G: 2D tensor with shape (k, n) where k is message length.
    Returns:
    - c: 2D int32 tensor of codewords (values 0 or 1), shape (num_blocks, n)
    """

    # handle 1D input: trim remainder and reshape into rows of length k
    if b.shape.rank == 1:
        G_shape = tf.shape(G)
        k = G_shape[0]
        L = tf.shape(b)[0]
        rem = tf.math.floormod(L, k)
        L2 = L - rem  # length we keep (may be 0)
        # slice first L2 elements (works in graph & eager)
        b_trimmed = tf.slice(b, [0], [L2])
        b2 = tf.reshape(b_trimmed, tf.stack([-1, k]))
    else:
        # assume already shaped (num_blocks, k)
        b2 = b

    # cast to an integer type for matmul; int32 is fine on CPU/GPU
    b2 = tf.cast(b2, tf.int32)
    G = tf.cast(G, tf.int32)

    # matrix multiply and reduce modulo 2 (GF(2) arithmetic)
    c = tf.math.floormod(tf.matmul(b2, G), 2)

    return tf.cast(c, tf.int32)


class LinearBlockEncoder(tf.keras.layers.Layer):
    '''
    LinearBlockEncoder class
    G: Generator Matrix
    '''

    def __init__(self, G, **kwargs):
        super().__init__(**kwargs)
        # store generator as a constant tensor (not trainable)
        self.G = tf.cast(tf.constant(G), tf.int32)

    def call(self, inputs):
        return encoder_tf(inputs, self.G)


def bp_decoder(llr, H, it, mode):
    '''Soft decoding for given code reflected by parity check matrix H by belief propagation
    llr:    Log-likelihood ratios
    H:      Parity check matrix
    it:     Number of iterations
    mode:   Exact (0, default) and approximate (1) calculation of boxplus
    '''
    bp_out = llr
    cv = 0
    for _ in range(0, it):
        vc = bp_out[:, :, np.newaxis] * H.T[np.newaxis, :, :] - cv
        cv = boxplus(vc, H, mode)
        bp_out = np.sum(cv, axis=-1) + llr

    cr = (np.sign(bp_out) < 0) * 1
    k = np.size(H, 1) - np.size(H, 0)
    br = cr[:, 0: k]
    return bp_out, cr, br


def boxplus(llrs, H, mode):
    '''Calculate boxplus operation of llrs with parity check matrix H
    mode: 0: boxplus / 1: boxplus approximation
    '''
    if mode == 1:
        H_uncon = (H == 0) * 1
        sign = np.prod(np.sign(llrs) + np.expand_dims(H_uncon.T, 0), axis=1,
                       keepdims=True) / np.sign(llrs + H_uncon.T[np.newaxis, :, :])
        mask = (np.ones((H.shape[-1], H.shape[-1])) - np.eye(H.shape[-1]))
        masked_llrs = np.transpose(np.abs(
            llrs)[:, :, :, np.newaxis] * mask[np.newaxis, :, np.newaxis, :], (0, 3, 2, 1))
        masked_llrs2 = np.ma.masked_equal(masked_llrs, 0.0, copy=False)
        mini = np.array(np.min(masked_llrs2, axis=-1)) * \
            H.T[np.newaxis, :, :]  # accurate
        res = sign * mini
    # elif mode == 2:
    #     # alternative more compact but slow implementation: at least 4 times slower
    #     mask = (np.ones((H.shape[-1], H.shape[-1])) - np.eye(H.shape[-1]))
    #     masked_llrs = np.transpose(llrs[: , :, :, np.newaxis] * mask[np.newaxis, :, np.newaxis, :], (0, 3, 2, 1))
    #     masked_llrs2 = np.ma.masked_equal(masked_llrs, 0.0, copy = False)
    #     vc_tanh = np.tanh(np.clip(masked_llrs2 / 2, -1e12, 1e12))
    #     cv = np.array(np.prod(vc_tanh, axis = -1)) * H.T[np.newaxis, :, :]
    #     cv = np.clip(np.array(cv), -1 + 1e-12, 1 - 1e-12)
    #     res = 2 * np.arctanh(cv)
    else:
        H_uncon = (H == 0) * 1
        vc_tanh = np.tanh(np.clip(llrs / 2, -1e12, 1e12))
        vc_tanh_prod = vc_tanh + H_uncon.T[np.newaxis, :, :]
        cv = np.prod(vc_tanh_prod, 1, keepdims=True)
        cv = (cv / vc_tanh_prod) * H.T[np.newaxis, :, :]
        cv = np.clip(np.array(cv), -1 + 1e-12, 1 - 1e-12)
        res = 2 * np.arctanh(cv)
    return res


def mimo_coding(c, Nt, M, arch):
    '''Encode code words c horizontally or vertically into c2
    INPUT
    c: code words of dim (Nbc, n)
    Nt: Number of transmit symbols
    M: Modulation order
    arch: PAC: Per antenna coding (horizontal) / PSC: Per stream coding (vertical)
    OUTPUT
    c2: MIMO encoding of c of dim (Nbc / n * log2(M), Nt, n / log2(M), log2(M))
    '''
    if arch == 'horiz':
        rest = np.mod(c.shape[-1], np.log2(M))
        if rest != 0:
            c_end = np.random.randint(
                2, size=(c.shape[0], int(np.log2(M) - rest)))
            c0 = np.concatenate((c, c_end), axis=-1)
        else:
            c0 = c
        c1 = np.reshape(c0, (-1, int(Nt * np.log2(M)), c0.shape[-1]))
        c2 = np.reshape(c1, (c1.shape[0], c1.shape[1], int(
            c1.shape[-1] / np.log2(M)), int(np.log2(M))))
    elif arch == 'vert':
        fit2x = Nt * np.log2(M) / c.shape[-1]
        if int(fit2x) >= 1:
            c0 = c.reshape((-1, int(c.shape[-1] * int(fit2x))))
            c_end = np.random.randint(
                2, size=(c0.shape[0], int(Nt * np.log2(M) - c0.shape[-1])))
            c1 = np.concatenate((c0, c_end), axis=-1)
            # expand dims by 1 for same processing
            c2 = c1.reshape(
                (c1.shape[0], int(c1.shape[1] / np.log2(M)), 1, int(np.log2(M))))
        else:
            c_end = np.random.randint(2, size=(c.shape[0], int(
                Nt * np.log2(M) * np.ceil(1 / fit2x) - c.shape[-1])))
            c0 = np.concatenate((c, c_end), axis=-1)
            c1 = c0.reshape(
                (-1, int(np.ceil(1 / fit2x)), int(Nt * np.log2(M))))
            c2 = np.transpose(c1.reshape((c1.shape[0], c1.shape[1], int(
                c1.shape[-1] / np.log2(M)), int(np.log2(M)))), (0, 2, 1, 3))
    else:
        print('Architecture not available.')
        c = c2
    return c2


def mimo_decoding(llr_c, n, Nt, M, arch):
    '''Decode horizontally or vertically encoded code word llrs [llr_c] of equalizer dimensions back into original [llr_c2]
    INPUT
    llr_c: LLRs of code bits from equalizer of dim (Nb, Nt, log2(M))
    n: code word length
    Nt: Number of transmit symbols
    M: Modulation order
    arch: PAC: Per antenna coding (horizontal) / PSC: Per stream coding (vertical)
    OUTPUT
    llr_c2: Original order of llr_c of dim (Nbc, n)
    '''
    if arch == 'horiz':
        llr_c1 = np.transpose(np.reshape(np.transpose(llr_c, (1, 0, 2)),
                                         (llr_c.shape[-2], -1, int(n + np.mod(np.log2(M) - n, np.log2(M))))), (1, 0, 2))
        llr_c2 = np.reshape(llr_c1, (-1, llr_c1.shape[-1]))[:, :n]
    elif arch == 'vert':
        fit2x = Nt * np.log2(M) / n
        if int(fit2x) >= 1:
            llr_c0 = llr_c.reshape((llr_c.shape[0], -1))
            llr_c1 = llr_c0[:, :int(
                llr_c0.shape[-1] - np.mod(llr_c0.shape[-1], n))]
            llr_c2 = llr_c1.reshape((-1, n))
        else:
            llr_c1 = llr_c.reshape(
                (-1, int(Nt * np.log2(M) * np.ceil(1 / fit2x))))
            llr_c2 = llr_c1[:, :n]
    else:
        print('Architecture not available.')
        llr_c2 = llr_c
    return llr_c2


def gpu_select(number=0, memory_growth=True, cpus=0):
    '''Select/deactivate GPU in Tensorflow 2
    Configure to use only a single GPU and allocate only as much memory as needed
    For more details, see https://www.tensorflow.org/guide/gpu
    '''
    if number >= 0:
        # Choose GPU
        gpus = tf.config.list_physical_devices('GPU')
        print('Number of GPUs available :', len(gpus))
        if gpus:
            gpu_number = number  # Index of the GPU to use
            try:
                tf.config.set_visible_devices(gpus[gpu_number], 'GPU')
                print('Only GPU number', gpu_number, 'used.')
                tf.config.experimental.set_memory_growth(
                    gpus[gpu_number], memory_growth)
            except RuntimeError as error:
                print(error)
    elif number == -1:
        # Deactivate GPUs and use CPUs
        try:
            tf.config.experimental.set_visible_devices([], 'GPU')
            print('GPUs deactivated.')
        except RuntimeError as error:
            print(error)
        if cpus > 0:
            try:
                tf.config.threading.set_intra_op_parallelism_threads(cpus)
                tf.config.threading.set_inter_op_parallelism_threads(1)
                print(cpus, 'CPUs used.')
            except RuntimeError as error:
                print(error)
    else:
        print('Will choose GPU or CPU automatically.')

# ------ Legacy ---------------------------


class CMDNetBinaryLegacy(tf.keras.Model):
    '''Binary CMDNet layer
    '''

    def __init__(self, it, m, alpha, taui0=1, delta0=1):
        super(CMDNetBinaryLegacy, self).__init__()
        self.it = tf.constant(it)
        self.m = m
        self.M = m.shape[0]
        # if (m == np.array([-1, 1])).all():
        alpha = alpha[:, 0]
        # else:
        #    alpha = alpha[:, 1]
        self.alphat = tf.constant(value=alpha)
        # func_tau = tau0 * np.ones(self.it + 1)
        self.taui = self.add_weight(shape=(it + 1,),
                                    initializer=tf.keras.initializers.Constant(
                                        value=taui0),
                                    trainable=True)
        self.delta = self.add_weight(shape=(it,),
                                     initializer=tf.keras.initializers.Constant(
                                         value=delta0),
                                     trainable=True)
        self.s0 = tf.constant(value=np.zeros_like(alpha))
        # self.G0 = self.add_weight(shape = alpha.shape,
        #                      initializer=tf.keras.initializers.Constant(value = np.zeros_like(alpha)),
        #                      trainable=False)
        # self.taui = tf.Variable(initial_value = 1 / func_tau,
        #                         trainable = True)
        # self.delta = tf.Variable(initial_value = delta0 * np.ones(it),
        #                          trainable = True)
        # b_init = tf.zeros_initializer()
        # self.b = tf.Variable(initial_value=b_init(shape=(units,),
        #                                          dtype='float32'),
        #                     trainable=True)

    # @tf.function#(jit_compile = True)
    def call(self, inputs):
        [yt, Ht, sigmat0] = inputs
        sigmat = tf.expand_dims(sigmat0, axis=-1)

        alphat = self.alphat
        s = KB.transpose(KB.expand_dims(self.s0)) * KB.ones_like(Ht[:, 0, :])
        taui_abs = KB.abs(self.taui[0])
        xt = KB.tanh((KB.log(1 / alphat - 1) + s) / 2 * taui_abs)

        # UNFOLDING
        HH = KB.batch_dot(KB.permute_dimensions(Ht, (0, 2, 1)), Ht)
        yH = KB.batch_dot(yt, Ht)
        # HTy = tf.squeeze(tf.matmul(tf.transpose(Ht, [0, 2, 1]), tf.expand_dims(yt, axis = -1)), axis = -1)
        # HTH = tf.matmul(tf.transpose(Ht, [0, 2, 1]), Ht)
        for iteration in tf.range(0, self.it):
            xHH = KB.batch_dot(xt, HH)
            grad_x = 1 / 2 * taui_abs * (1 - xt ** 2)
            grad_L = sigmat ** 2 * KB.tanh(s / 2) + grad_x * (xHH - yH)
            # grad_L = KB.tanh(s / 2) + 1 / sigmat ** 2 * grad_x * (xHH - yH) # original version
            # Gradient/ResNet Layer
            s = s - self.delta[iteration] * grad_L

            # Start of new iteration
            # no negative values for tau !
            taui_abs = KB.abs(self.taui[iteration + 1])
            xt = KB.tanh((KB.log(1 / alphat - 1) + s) / 2 * taui_abs)
            xt2 = KB.expand_dims(xt, axis=-1)
            # [q(x = m_1), q(x = m_2)]
            ft = tf.concat([(1 + self.m[0] * xt2) / 2,
                           (1 + self.m[1] * xt2) / 2], axis=-1)
        return ft, xt
