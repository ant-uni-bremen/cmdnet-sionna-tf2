#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 01 14:33:08 2022

@author: beck
"""

# LOADED PACKAGES
# Tensorflow packages
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
from unittest import findTestCases
from sklearn.metrics import fowlkes_mallows_score
import tensorflow as tf
from tensorflow.keras import layers
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0 # Number of the GPU to be used
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)

# Import Sionna
try:
    import sionna as sn
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna as sn
from sionna.utils import BinarySource, QAMSource, ebnodb2no, compute_ser, compute_ber, PlotBER
from sionna.channel import FlatFadingChannel, KroneckerModel
from sionna.channel.utils import exp_corr_mat
from sionna.mimo import lmmse_equalizer
from sionna.mapping import SymbolDemapper, Mapper, Demapper, Constellation
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from tensorflow.keras import backend as KB
import tikzplotlib as tplt
from tensorflow.keras.models import Model
import time
from sionna.utils.metrics import BitwiseMutualInformation
# import myfunctions as mf



sn.config.xla_compat = True
class Model(tf.keras.Model):
    def __init__(self, algo, spatial_corr = None, num_tx_ant = 32, num_rx_ant = 32, const = Constellation('pam', 1),
                 code = False, n = 1024, k = 512, code_it = 10, code_train = False, trainbit = False):
        super().__init__()
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant
        self.code = code
        self.trainbit = trainbit
        self.algo = algo # choose algorithm, change in self.algo inactive with tf.function decorator
        self.binary_source = BinarySource()
        self.num_bits_per_symbol = const._num_bits_per_symbol
        self.mod = const._constellation_type
        # self.constellation = const # Constellation(mod, num_bits_per_symbol, trainable = False)
        self.mapper = Mapper(constellation = const)
        # self.demapper = Demapper("app", constellation = self.constellation)
        self.channel = FlatFadingChannel(self.num_tx_ant,
                                         self.num_rx_ant,
                                         spatial_corr = spatial_corr,
                                         add_awgn = True,
                                         return_channel = True)
        # self.val_batch_size = 10000
        if self.code == True:
            self.n = n
            self.k = k
            self.coderate = self.k / self.n
            self.encoder = LDPC5GEncoder(self.k, self.n)
            self.decoder = LDPC5GDecoder(self.encoder,
                                         num_iter = code_it,
                                         trainable = code_train,
                                         hard_out = False)
            ## Own BP implementation
            # saveobj = mf.savemodule('npz')
            # code = saveobj.load(os.path.join('codes', 'LDPC64x128'))
            # self.G = code['G']
            # self.H = code['H']
            ## c = mf.encoder(b, self.G)
            ## [llr_b, c_hat, b_hat] = mf.bp_decoder(llr_c, self.H, 10, 0)
        else:
            self.coderate = 1
    @tf.function#(jit_compile = True)
    def call(self, batch_size, ebno_db, ebno_db_max = None):
        
        # Foward model
        if self.code == True:
            b = self.binary_source([batch_size, self.num_tx_ant, self.k])
            c = self.encoder(b)
            ## Own BP implementation
            # c = mf.encoder(b, self.G)
        else:
            b = self.binary_source([batch_size, self.num_tx_ant])
            c = b 
        shapec = tf.shape(c)

        x = self.mapper(c)
        # shape = x.shape
        x = tf.reshape(x, [-1, self.num_tx_ant])
        
        if ebno_db_max == None:
            ebno_db_max = ebno_db
        ebno_db_vec = tf.random.uniform(shape = [tf.shape(x)[0]], minval = ebno_db, maxval = ebno_db_max)
        if self.mod == 'pam':
            no = 1 / 2 * ebnodb2no(ebno_db_vec, self.num_bits_per_symbol, self.coderate)
        else:
            no = ebnodb2no(ebno_db_vec, self.num_bits_per_symbol, self.coderate)
        no_scale = self.num_rx_ant  # scaling of noise variance necessary since channel has Rayleigh taps with variance 1
        # no *= np.sqrt(self.num_rx_ant)    # original, why square root?
        y, h = self.channel([x, no * no_scale])
        sigmat0 = tf.math.sqrt(no)

        # Receiver side
        ft, x_hat, llr_c, p_c = self.algo([y, h, sigmat0])
        
        llr_c = tf.reshape(llr_c, shapec)
        p_c = tf.reshape(p_c, shapec)
        
        if self.code == True:
            llr_b = self.decoder(llr_c)
            ## Own BP implementation
            # shape_llr = llr_c.shape
            # llr_c = tf.reshape(llr_c, [-1, 128])
            # llr_hat, _, _ = mf.bp_decoder(llr_c, self.H, 10, 0)
            # llr_b = tf.reshape(llr_hat[..., 0:64], [shape_llr[0], shape_llr[1], -1])
            #
            symprob2llr = tf_symprob2llr(2, b = 1)
            p_b = symprob2llr.llr2prob(llr_b)
        else:
            llr_b = llr_c
            p_b = p_c
        b_hat = tf.cast(llr_b > 0, dtype = 'float32')

        # Training objective function
        if self.trainbit:
            bce = tf.keras.losses.BinaryCrossentropy()
            loss = bce(b, p_b)
        else:
            soft = 0
            if soft == 1:
                # 2. mean should be a sum for overall MSE scaling with Nt
                if not mod == 'pam':
                    x2 = tf.concat([tf.math.real(x), tf.math.imag(x)], axis = -1)
                else:
                    x2 = tf.cast(x, dtype = 'float32')
                loss = tf.reduce_mean(tf.reduce_mean((x2 - x_hat) ** 2, axis = -1))
            else:
                if self.algo.algo_name == 'MMSE':
                    loss = []
                else:
                    # 2. mean should be a sum since q factorizes
                    c_cl = tf.reshape(c, [-1, self.num_tx_ant, self.num_bits_per_symbol])
                    cl = tf_bin2int(c_cl, axis = -1)
                    if self.mod == 'pam':
                        cl = tf_bin2int(c_cl, axis = -1)
                        M = 2 ** self.num_bits_per_symbol
                    else:
                        # e.g., c = 101010 -> c_re = 111, c_im = 000
                        c_cl_re = tf_bin2int(c_cl[..., ::2], axis = -1)
                        c_cl_im = tf_bin2int(c_cl[..., 1::2], axis = -1)
                        # concatenate classes of real and imaginary part as in equivalent real-valued system model
                        cl = tf.concat([c_cl_re, c_cl_im], axis = -1)
                        M = 2 ** int(self.num_bits_per_symbol / 2)
                    z = tf.one_hot(cl, depth = M)
                    loss = tf.reduce_mean(tf.reduce_mean(tf.keras.losses.categorical_crossentropy(z, ft, axis = -1), axis = -1))
        return b, b_hat, loss


@tf.function#(jit_compile = True)
def tf_bin2int(b, axis = -1, dtype = 'int64'):
    '''Tensorflow version: Convert a N-bit bit vector [b] across dimension [axis] into positive integer num [cl]
    Maximum integer number with uint64 is 9223372036854775807
    '''
    b_power = 2 ** tf.range(b.shape[axis], dtype = b.dtype)[::-1]
    cl = tf.cast(tf.reduce_sum(b * b_power, axis = axis), dtype = dtype)
    return cl

@tf.function#(jit_compile = True)
def tf_int2bin(x, N):
    '''Tensorflow version: Convert a positive integer num into an N-bit bit vector
    Limited up to N = 64 bits and 2 ** 64 numbers (!!!)
    '''
    return tf.math.mod(tf.bitwise.right_shift(tf.expand_dims(x, 1), tf.range(N, dtype = x.dtype)), 2)


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
    print_time = "{}:{:02d}:{:02d}:{:02d}".format(-int(d), int(h), int(m), int(np.round(s)))
    return print_time


class tf_symprob2llr():
    '''Convert symbol probabilities to llrs and probabilities of each transmitted bit
    Note: Gray Mapping assumed for bit to symbol mapping
    '''
    # Class Attribute
    name = 'Tensorflow version of symprob2llr'
    # Initializer / Instance Attributes
    def __init__(self, M, b = 0):
        '''
        M: modulation order
        b: bit reference, LLR = p(b = 0) / p(b = 1) vs. LLR = p(b = 1) / p(b = 0)
        '''
        self.M = M
        self.c_poss = tf_int2bin(np.array(range(0, M)), int(np.log2(M)))
        self.mask = tf.cast((self.c_poss == b)[np.newaxis, np.newaxis, ...], dtype = 'float32')
    # Instance methods
    @tf.function#(jit_compile = True)
    def __call__(self, p_m):
        '''
        p_m: symbol probability
        llr_c: log-likelihood ratio of bit c
        p_c: bit probability of c being b = 0 or b = 1 according to bit reference
        '''
        eps = tf.constant(1e-20)
        p_c = tf.reduce_sum(p_m[..., tf.newaxis] * self.mask, axis = -2)
        llr_c = tf.math.log(p_c / (1 - p_c + eps))   # avoid NaN
        llr_c = tf.clip_by_value(llr_c, -1e9, 1e9)   # avoid infinity
        return llr_c, p_c
    @tf.function#(jit_compile = True)
    def llr2prob(self, llr):
        '''
        Convert bit (!) llr into bit probability
        llr: log-likelihood ratio
        p_c: bit probability
        '''
        llr = tf.clip_by_value(llr, -90, 20)   # avoid NaN, for llr -> +infty: infty/infty = NaN
        p_c = tf.math.exp(llr) / (tf.math.exp(llr) + 1)
        return p_c


class algo_mmse(tf.keras.Model):
    """The MMSE algorithm with llr output
    """
    def __init__(self, constellation,
                    **kwargs):
        super(algo_mmse, self).__init__(**kwargs)
        self.algo_name = 'MMSE'
        self.demapper = Demapper('app', constellation = constellation)
        self.mod = constellation._constellation_type
        if self.mod == 'pam':
            self.M = constellation.points.shape[0]
        else:
            self.M = int(constellation.points.shape[0] / 2)
        self.symprob2llr = tf_symprob2llr(self.M, b = 1)

    @tf.function#(jit_compile = True)
    def call(self, inputs):
        [y, h, sigmat0] = inputs
        no = sigmat0 ** 2
        num_rx_ant = h.shape[-2]
        num_tx_ant = h.shape[-1]
        y_scale = np.sqrt(2 / num_rx_ant)
        if self.mod == 'pam':
            Ht = y_scale * tf.cast(tf.math.real(h), dtype = 'complex64')
            yt = y_scale * tf.cast(tf.math.real(y), dtype = 'complex64')
            s = tf.complex(no[..., tf.newaxis, tf.newaxis] * tf.eye(num_rx_ant, num_rx_ant), 0.0)
        else:
            compl = 2
            if compl == 1:
                # Complex: Version 1
                Ht = np.sqrt(1 / num_rx_ant) * h
                yt = np.sqrt(1 / num_rx_ant) * y
                s = tf.complex(no[..., tf.newaxis, tf.newaxis] * tf.eye(num_rx_ant, num_rx_ant), 0.0)
            elif compl == 2:
                # Complex: Version 2 without scaling
                no2 = no * num_rx_ant
                Ht = h
                yt = y
                s = tf.complex(no2[..., tf.newaxis, tf.newaxis] * tf.eye(num_rx_ant, num_rx_ant), 0.0)
            else:
                # Real-valued
                # H_scale = np.sqrt(2 / (2 * num_rx_ant)) # Not used in combination with demapper
                Ht = y_scale * tf.concat([tf.concat([tf.math.real(h), -tf.math.imag(h)], axis = -1), tf.concat([tf.math.imag(h), tf.math.real(h)], axis = -1)], axis = 1)
                yt = y_scale * tf.concat([tf.math.real(y), tf.math.imag(y)], axis = -1)
                s = tf.complex(no[..., tf.newaxis, tf.newaxis] * tf.eye(2 * num_rx_ant, 2 * num_rx_ant), 0.0)

        x_hat, no_eff = lmmse_equalizer(tf.cast(yt, dtype = 'complex64'), tf.cast(Ht, dtype = 'complex64'), s)
        # demapper expects complex x_hat normalized w.r.t. Re and Im part (!!!)
        if self.mod == 'qam' and compl == 0:
            x_hat = x_hat[:, :num_tx_ant] + 1j * x_hat[:, num_tx_ant:]
            no_eff = no_eff[:, :num_tx_ant] + no_eff[:, num_tx_ant:]
        # x_hat = tf.reshape(x_hat, shape)
        # no_eff = tf.reshape(no_eff, shape)
    
        llr_c = self.demapper([tf.cast(x_hat, dtype = 'complex64'), tf.cast(no_eff, dtype = 'float32')])
        p_c = self.symprob2llr.llr2prob(llr_c)
        ft = []
        return ft, x_hat, llr_c, p_c


class algo_cmdnet(tf.keras.Model):
    """The CMDNet algorithm with llr output
    """
    def __init__(self, num_iter = 64, const = Constellation('pam', 1), num_tx_ant = 1, binary = False, taui0 = 1, delta0 = 1,
                    **kwargs):
        super(algo_cmdnet, self).__init__(**kwargs)
        self.algo_name = 'CMDNet'
        num_bits_per_symbol = const._num_bits_per_symbol
        self.mod = const._constellation_type
        m_mapper = const.points
        self.binary = binary
        M = m_mapper.shape[0]
        num_bits_per_symbol = int(np.log2(M))


        if self.mod == 'pam':
            M = 2 ** num_bits_per_symbol
            m = tf.math.real(m_mapper) # self.mapper.constellation.points
            alpha = 1 / M * np.ones((num_tx_ant, M), dtype = 'float32')
        else:
            M = 2 ** int(num_bits_per_symbol / 2)
            c_poss = tf_int2bin(np.array(range(0, 2 ** int(num_bits_per_symbol / 2))), int(num_bits_per_symbol / 2))
            b_power = 2 ** np.array(range(0, num_bits_per_symbol))[::-1][::2]
            ind_m = np.sum(b_power * c_poss, axis = -1)
            m = np.sqrt(2) * tf.math.real(tf.gather(m_mapper, ind_m, axis = -1))
            alpha = 1 / M * np.ones((2 * num_tx_ant, M), dtype = 'float32')
            ## Old index calculation for QAM modulation:
            # c_poss = tf_int2bin(np.array(range(0, 2 ** num_bits_per_symbol)), num_bits_per_symbol)
            # ind_m = mf.bin2int(c_poss[(c_poss[:, 1::2] == 0).all(axis = -1)], dtype = 'int64')
        
        if self.binary == True:
            self.bpsk = False
            self.cmdnet = cmdnet_bin(num_iter, m, alpha, bpsk = self.bpsk, taui0 = taui0, delta0 = delta0)
        else:
            self.cmdnet = cmdnet(num_iter, m, alpha, taui0 = taui0, delta0 = delta0)
        self.symprob2llr = tf_symprob2llr(M, b = 1)

    @tf.function#(jit_compile = True)
    def call(self, inputs):
        [y, h, sigmat0] = inputs
        num_rx_ant = h.shape[-2]
        num_tx_ant = h.shape[-1]
        y_scale = np.sqrt(2 / num_rx_ant)
        # Wrapper
        if self.mod == 'pam':
            Ht = y_scale * tf.math.real(h)
            yt = y_scale * tf.math.real(y)
        else:
            H_scale = np.sqrt(2 / (2 * num_rx_ant))
            Ht = H_scale * tf.concat([tf.concat([tf.math.real(h), -tf.math.imag(h)], axis = -1), tf.concat([tf.math.imag(h), tf.math.real(h)], axis = -1)], axis = 1)
            yt = y_scale * tf.concat([tf.math.real(y), tf.math.imag(y)], axis = -1)
        # Real-valued
        ft, xt = self.cmdnet([yt, Ht, sigmat0])
        llr_c, p_c = self.symprob2llr(ft)
        if not self.mod == 'pam':
            # llr_c = llr_c[..., tf.newaxis]
            # p_c = p_c[..., tf.newaxis]
            llr_c = tf.concat([llr_c[:, :num_tx_ant], llr_c[:, num_tx_ant:]], axis = -1)
            llr_c = tf.concat([llr_c[..., 0::2], llr_c[..., 1::2]], axis = -1)
            p_c = tf.concat([p_c[:, :num_tx_ant], p_c[:, num_tx_ant:]], axis = -1)
            p_c = tf.concat([p_c[..., 0::2], p_c[..., 1::2]], axis = -1)
            # ft = tf.reshape(tf.concat([ft[:, :model1.num_tx_ant][..., tf.newaxis], ft[:, model1.num_tx_ant:][..., tf.newaxis]], axis = -1), [-1, ft.shape[1], ft.shape[2]])
        return ft, xt, llr_c, p_c


class cmdnet(tf.keras.Model):
    """The CMDNet algorithm
    """
    def __init__(self, num_iter = 64, m = np.array([-1, 1]), alpha = np.array([0.5, 0.5]), taui0 = 1, delta0 = 1,
                    **kwargs):
        super(cmdnet, self).__init__(**kwargs)
        self.it = tf.constant(num_iter)
        # TODO: m and alpha trainable? -> needs to be changed in constellation object
        # and given to layers/function call
        self.mt = tf.expand_dims(tf.expand_dims(m, axis = 0), axis = 0)
        self.alpha = tf.constant(value = alpha)
        self.M = m.shape[0]
        taui0 = tf.constant(taui0, shape = (num_iter + 1), dtype = 'float32')
        delta0 = tf.constant(delta0, shape = (num_iter), dtype = 'float32')
        self.G0 = tf.constant(value = np.zeros_like(alpha))
        self.cmd_layer = []
        self.taui = []
        self.delta = []
        for ii in range(num_iter + 1):
            if ii == 0:
                self.cmd_layer.append(cmdnet_layer(m = self.mt, alpha = self.alpha, taui = taui0[ii], delta = delta0[ii], first_iter = True))
            elif ii == num_iter:
                self.cmd_layer.append(cmdnet_layer(m = self.mt, alpha = self.alpha, taui = taui0[ii], delta = 0, last_iter = True))
            else:
                self.cmd_layer.append(cmdnet_layer(m = self.mt, alpha = self.alpha, taui = taui0[ii], delta = delta0[ii]))
            self.taui.append(self.cmd_layer[-1].taui)
            if not ii == num_iter:
                self.delta.append(self.cmd_layer[-1].delta)

    @tf.function#(jit_compile = True)
    def call(self, inputs):
        [yt, Ht, sigmat0] = inputs
        HH = tf.matmul(Ht, Ht, transpose_a = True)
        yH = tf.squeeze(tf.matmul(tf.expand_dims(yt, axis = 1), Ht), axis = 1)
        G = tf.expand_dims(self.G0, axis = 0) * tf.expand_dims(tf.ones_like(Ht[:, 0, :]), axis = -1)
        sigmat = tf.expand_dims(tf.expand_dims(sigmat0, axis = -1), axis = -1)
        for layer in self.cmd_layer:
            G, ft, xt = layer(G, HH, yH, sigmat)
        return ft, xt


class cmdnet_layer(tf.keras.layers.Layer):
    """The CMDNet layer
    """
    def __init__(self, m = np.array([-1, 1]), alpha = np.array([0.5, 0.5]), taui = 1, delta = 1, first_iter = False, last_iter = False):
        super().__init__()
        # TODO: m and alpha trainable? -> needs to be changed in constellation object
        # and given to layers/function call
        self.mt = tf.constant(m)
        self.alpha = tf.constant(value = alpha)
        self.first_iter = first_iter
        self.last_iter = last_iter
        self.taui = self.add_weight(shape = (1,),
                             initializer = tf.keras.initializers.Constant(value = taui),
                             trainable = True,
                             name = 'taui')
        if not self.last_iter:
            self.delta = self.add_weight(shape = (1,),
                                initializer = tf.keras.initializers.Constant(value = delta),
                                trainable = True,
                                name = 'delta')

    @tf.function#(jit_compile = True)
    def call(self, G, HH, yH, sigmat):
        # Start of new iteration
        taui_abs = tf.math.abs(self.taui) # no negative values for tau !
        if self.first_iter == True:
            ft = tf.nn.softmax((tf.math.log(self.alpha) + G) * 1, axis = -1)
        else:
            ft = tf.nn.softmax((tf.math.log(self.alpha) + G) * taui_abs, axis = -1)
        xt = tf.reduce_sum(ft * self.mt, axis = -1)
        
        if not self.last_iter:
            xHH = tf.squeeze(tf.matmul(tf.expand_dims(xt, axis = 1), HH), axis = 1)
            grad_x = taui_abs * (ft * self.mt - ft * tf.expand_dims(xt, axis = -1))
            # grad_L =  (1 - tf.math.exp(-G)) + 1 / sigmat ** 2 * grad_x * tf.expand_dims(xHH - yH, axis = -1) # original version
            grad_L = sigmat ** 2 * (1 - tf.math.exp(-G)) + grad_x * tf.expand_dims(xHH - yH, axis = -1)
            # Gradient/ResNet Layer
            G = G - self.delta * grad_L
        return G, ft, xt


# class cmdnet_bin2(tf.keras.Model):
#     '''Binary CMDNet layer
#     '''
#     def __init__(self, it, m, alpha, taui0 = 1, delta0 = 1):
#         super(cmdnet_bin2, self).__init__()
#         self.it = tf.constant(it)
#         self.m = m
#         self.M = m.shape[0]
#         # if (m == np.array([-1, 1])).all():
#         alpha = alpha[:, 0]
#         #else:
#         #    alpha = alpha[:, 1]
#         self.alphat = tf.constant(value = alpha)
#         #func_tau = tau0 * np.ones(self.it + 1)
#         self.taui = self.add_weight(shape = (it + 1,),
#                              initializer = tf.keras.initializers.Constant(value = taui0),
#                              trainable = True)
#         self.delta = self.add_weight(shape = (it,),
#                              initializer = tf.keras.initializers.Constant(value = delta0),
#                              trainable = True)
#         self.s0 = tf.constant(value = np.zeros_like(alpha))
#         # self.G0 = self.add_weight(shape = alpha.shape,
#         #                      initializer=tf.keras.initializers.Constant(value = np.zeros_like(alpha)),
#         #                      trainable=False)
#         # self.taui = tf.Variable(initial_value = 1 / func_tau,
#         #                         trainable = True)
#         # self.delta = tf.Variable(initial_value = delta0 * np.ones(it),
#         #                          trainable = True)
#         #b_init = tf.zeros_initializer()
#         #self.b = tf.Variable(initial_value=b_init(shape=(units,),
#         #                                          dtype='float32'),
#         #                     trainable=True)
    
#     #@tf.function#(jit_compile = True)
#     def call(self, inputs):
#         [yt, Ht, sigmat0] = inputs
#         sigmat = tf.expand_dims(sigmat0, axis = -1)
        
#         alphat = self.alphat
#         s = KB.transpose(KB.expand_dims(self.s0)) * KB.ones_like(Ht[:, 0, :])
#         taui_abs = KB.abs(self.taui[0])
#         xt = KB.tanh((KB.log(1 / alphat - 1) + s) / 2 * taui_abs)
        
#         # UNFOLDING
#         HH = KB.batch_dot(KB.permute_dimensions(Ht, (0, 2, 1)), Ht)
#         yH = KB.batch_dot(yt, Ht)
#         # HTy = tf.squeeze(tf.matmul(tf.transpose(Ht, [0, 2, 1]), tf.expand_dims(yt, axis = -1)), axis = -1)
#         # HTH = tf.matmul(tf.transpose(Ht, [0, 2, 1]), Ht)
#         for iteration in tf.range(0, self.it):
#             xHH = KB.batch_dot(xt, HH)
#             grad_x = 1 / 2 * taui_abs * (1 - xt ** 2) 
#             grad_L = sigmat ** 2 * KB.tanh(s / 2) + grad_x * (xHH - yH)
#             # grad_L = KB.tanh(s / 2) + 1 / sigmat ** 2 * grad_x * (xHH - yH) # original version
#             # Gradient/ResNet Layer
#             s = s - self.delta[iteration] * grad_L
            
#             # Start of new iteration
#             taui_abs = KB.abs(self.taui[iteration + 1]) # no negative values for tau !
#             xt = KB.tanh((KB.log(1 / alphat - 1) + s) / 2 * taui_abs)
#             xt2 = KB.expand_dims(xt, axis = -1)
#             ft = tf.concat([(1 + self.m[0] * xt2) / 2, (1 + self.m[1] * xt2) / 2], axis = -1) # [q(x = m_1), q(x = m_2)]
#         return ft, xt


class cmdnet_bin(tf.keras.Model):
    """The binary CMDNet algorithm: bpsk or generic modulation
    """
    def __init__(self, num_iter = 64, m = np.array([1, -1]), alpha = np.array([0.5, 0.5]), taui0 = 1, delta0 = 1, bpsk = True,
                    **kwargs):
        super(cmdnet_bin, self).__init__(**kwargs)
        self.it = num_iter
        # TODO: m and alpha trainable? -> needs to be changed in constellation object
        # and given to layers/function call
        self.mt = m # tf.constant(m)
        #if (m == np.array([-1, 1])).all():
        #    alpha = alpha[:, 1]
        #else:
        alpha = alpha[:, 0]
        self.alpha = tf.constant(value = alpha)
        self.bpsk = bpsk
        self.M = m.shape[0]
        taui0 = tf.constant(taui0, shape = (num_iter + 1), dtype = 'float32')
        delta0 = tf.constant(delta0, shape = (num_iter), dtype = 'float32')
        
        self.s0 = tf.constant(value = np.zeros_like(alpha))
        self.cmd_layer = []
        self.taui = []
        self.delta = []
        for ii in range(num_iter + 1):
            if self.bpsk == True:
                if ii == 0:
                    self.cmd_layer.append(cmdnet_bin_layer(m = self.mt, alpha = self.alpha, taui = taui0[ii], delta = delta0[ii], first_iter = True))
                elif ii == num_iter:
                    self.cmd_layer.append(cmdnet_bin_layer(m = self.mt, alpha = self.alpha, taui = taui0[ii], delta = 0, last_iter = True))
                else:
                    self.cmd_layer.append(cmdnet_bin_layer(m = self.mt, alpha = self.alpha, taui = taui0[ii], delta = delta0[ii]))
            else:
                if ii == 0:
                    self.cmd_layer.append(cmdnet_bin_generic_layer(m = self.mt, alpha = self.alpha, taui = taui0[ii], delta = delta0[ii], first_iter = True))
                elif ii == num_iter:
                    self.cmd_layer.append(cmdnet_bin_generic_layer(m = self.mt, alpha = self.alpha, taui = taui0[ii], delta = 0, last_iter = True))
                else:
                    self.cmd_layer.append(cmdnet_bin_generic_layer(m = self.mt, alpha = self.alpha, taui = taui0[ii], delta = delta0[ii]))
            self.taui.append(self.cmd_layer[-1].taui)
            if not ii == num_iter:
                self.delta.append(self.cmd_layer[-1].delta)

    @tf.function#(jit_compile = True)
    def call(self, inputs):
        [yt, Ht, sigmat0] = inputs
        HH = tf.matmul(Ht, Ht, transpose_a = True)
        yH = tf.squeeze(tf.matmul(tf.expand_dims(yt, axis = 1), Ht), axis = 1)
        s = tf.transpose(tf.expand_dims(self.s0, axis = -1)) * tf.ones_like(Ht[:, 0, :])
        sigmat = tf.expand_dims(sigmat0, axis = -1)
        for layer in self.cmd_layer:
            s, ft, xt = layer(s, HH, yH, sigmat)
        return ft, xt



class cmdnet_bin_layer(tf.keras.layers.Layer):
    """The binary CMDNet layer
    """
    def __init__(self, m = np.array([1, -1]), alpha = np.array([0.5, 0.5]), taui = 1, delta = 1, first_iter = False, last_iter = False):
        super().__init__()
        # TODO: m and alpha trainable? -> needs to be changed in constellation object
        # and given to layers/function call
        self.m = m # tf.constant(m)
        self.alpha = tf.constant(alpha)
        self.first_iter = first_iter
        self.last_iter = last_iter
        self.taui = self.add_weight(shape = (1,),
                             initializer = tf.keras.initializers.Constant(value = taui),
                             trainable = True,
                             name = 'taui')
        if not self.last_iter:
            self.delta = self.add_weight(shape = (1,),
                                initializer = tf.keras.initializers.Constant(value = delta),
                                trainable = True,
                                name = 'delta')

    @tf.function#(jit_compile = True)
    def call(self, s, HH, yH, sigmat):
        # Start of new iteration
        taui_abs = tf.math.abs(self.taui) # no negative values for tau !
        if self.first_iter == True:
            xt = tf.math.tanh((tf.math.log(1 / self.alpha - 1) + s) / 2 * 1)
        else:
            xt = tf.math.tanh((tf.math.log(1 / self.alpha - 1) + s) / 2 * taui_abs)
        xt2 = tf.expand_dims(xt, axis = -1)
        # if tf.math.reduce_all(self.m == np.array([-1, 1])):
        #     ft = tf.concat([(1 - xt2) / 2, (1 + xt2) / 2], axis = -1) # [q(x = -1), q(x = 1)]
        # else:
        #     ft = tf.concat([(1 + xt2) / 2, (1 - xt2) / 2], axis = -1) # [q(x = 1), q(x = -1)]
        ft = tf.concat([(1 + self.m[0] * xt2) / 2, (1 + self.m[1] * xt2) / 2], axis = -1) # [q(x = m_1), q(x = m_2)]
        
        if not self.last_iter:
            xHH = tf.squeeze(tf.matmul(tf.expand_dims(xt, axis = 1), HH), axis = 1)
            grad_x = 1 / 2 * taui_abs * (1 - xt ** 2) 
            grad_L = sigmat ** 2 * tf.math.tanh(s / 2) + grad_x * (xHH - yH)
            # grad_L = tf.math.tanh(s / 2) + 1 / sigmat ** 2 * grad_x * (xHH - yH) # original version
            # Gradient/ResNet Layer
            s = s - self.delta * grad_L
        return s, ft, xt


class cmdnet_bin_generic_layer(tf.keras.layers.Layer):
    """The generic binary CMDNet layer: New general case where we have 2 arbitrary symbols in m
    """
    def __init__(self, m = np.array([1, -1]), alpha = np.array([0.5, 0.5]), taui = 1, delta = 1, first_iter = False, last_iter = False):
        super().__init__()
        # TODO: m and alpha trainable? -> needs to be changed in constellation object
        # and given to layers/function call
        self.m = tf.constant(m)
        self.alpha = tf.constant(alpha)
        self.first_iter = first_iter
        self.last_iter = last_iter
        self.taui = self.add_weight(shape = (1,),
                             initializer = tf.keras.initializers.Constant(value = taui),
                             trainable = True,
                             name = 'taui')
        if not self.last_iter:
            self.delta = self.add_weight(shape = (1,),
                             initializer = tf.keras.initializers.Constant(value = delta),
                             trainable = True,
                             name = 'delta')

    @tf.function#(jit_compile = True)
    def call(self, s, HH, yH, sigmat):
        # Start of new iteration
        taui_abs = tf.math.abs(self.taui) # no negative values for tau !
        if self.first_iter == True:
            ft0 = tf.math.sigmoid(-(tf.math.log(1 / self.alpha - 1) + s) * 1)
        else:
            ft0 = tf.math.sigmoid(-(tf.math.log(1 / self.alpha - 1) + s) * taui_abs)
        ft = tf.concat([ft0[..., tf.newaxis], (1 - ft0)[..., tf.newaxis]], axis = -1)
        xt = ft0 * (self.m[0] - self.m[1]) + self.m[1]

        if not self.last_iter:
            xHH = tf.squeeze(tf.matmul(tf.expand_dims(xt, axis = 1), HH), axis = 1)
            grad_x = -taui_abs * ft0 * (1 - ft0) * (self.m[0] - self.m[1])
            grad_L = sigmat ** 2 * tf.math.tanh(s / 2) + grad_x * (xHH - yH)
            # grad_L = tf.math.tanh(s / 2) + 1 / sigmat ** 2 * grad_x * (xHH - yH) # original version
            # Gradient/ResNet Layer
            s = s - self.delta * grad_L
        return s, ft, xt


class algo_amp(tf.keras.Model):
    """The AMP algorithm with llr output
    """
    def __init__(self, num_iter = 64, const = Constellation('pam', 1), num_tx_ant = 1, binary = False, gamma0 = 1, delta0 = 1,
                    **kwargs):
        super(algo_amp, self).__init__(**kwargs)
        self.algo_name = 'AMP'
        num_bits_per_symbol = const._num_bits_per_symbol
        self.mod = const._constellation_type
        m_mapper = const.points
        self.binary = binary
        M = m_mapper.shape[0]
        num_bits_per_symbol = int(np.log2(M))


        if self.mod == 'pam':
            M = 2 ** num_bits_per_symbol
            m = tf.math.real(m_mapper) # self.mapper.constellation.points
            alpha = 1 / M * np.ones((num_tx_ant, M), dtype = 'float32')
        else:
            M = 2 ** int(num_bits_per_symbol / 2)
            c_poss = tf_int2bin(np.array(range(0, 2 ** int(num_bits_per_symbol / 2))), int(num_bits_per_symbol / 2))
            b_power = 2 ** np.array(range(0, num_bits_per_symbol))[::-1][::2]
            ind_m = np.sum(b_power * c_poss, axis = -1)
            m =  np.sqrt(2) * tf.math.real(tf.gather(m_mapper, ind_m, axis = -1))
            alpha = 1 / M * np.ones((2 * num_tx_ant, M), dtype = 'float32')
        
        self.amp = amp(num_iter, m, alpha, binary = self.binary, gamma0 = gamma0, delta0 = delta0)
        self.symprob2llr = tf_symprob2llr(M, b = 1)

    @tf.function#(jit_compile = True)
    def call(self, inputs):
        [y, h, sigmat0] = inputs
        num_rx_ant = h.shape[-2]
        num_tx_ant = h.shape[-1]
        y_scale = np.sqrt(2 / num_rx_ant)
        # Wrapper
        if self.mod == 'pam':
            Ht = y_scale * tf.math.real(h)
            yt = y_scale * tf.math.real(y)
        else:
            H_scale = np.sqrt(2 / (2 * num_rx_ant))
            Ht = H_scale * tf.concat([tf.concat([tf.math.real(h), -tf.math.imag(h)], axis = -1), tf.concat([tf.math.imag(h), tf.math.real(h)], axis = -1)], axis = 1)
            yt = y_scale * tf.concat([tf.math.real(y), tf.math.imag(y)], axis = -1)
        # Real-valued
        ft, xt = self.amp([yt, Ht, sigmat0])
        llr_c, p_c = self.symprob2llr(ft)
        if not self.mod == 'pam':
            # llr_c = llr_c[..., tf.newaxis]
            # p_c = p_c[..., tf.newaxis]
            llr_c = tf.concat([llr_c[:, :num_tx_ant], llr_c[:, num_tx_ant:]], axis = -1)
            llr_c = tf.concat([llr_c[..., 0::2], llr_c[..., 1::2]], axis = -1)
            p_c = tf.concat([p_c[:, :num_tx_ant], p_c[:, num_tx_ant:]], axis = -1)
            p_c = tf.concat([p_c[..., 0::2], p_c[..., 1::2]], axis = -1)
            # ft = tf.reshape(tf.concat([ft[:, :model1.num_tx_ant][..., tf.newaxis], ft[:, model1.num_tx_ant:][..., tf.newaxis]], axis = -1), [-1, ft.shape[1], ft.shape[2]])
        return ft, xt, llr_c, p_c


class amp(tf.keras.Model):
    '''Approximate Message Passing Algorithm (AMP)
    Tensorflow implementation according to "Optimal Detection in Large MIMO" - Jeon et al., 2018, pp. 14 / 38
    for higher order modulation alphabets
    for binary modulation alphabet m = [-1, 1] / [1, -1]
    --------------------------------------------------
    INPUT
    inputs: list of [yt, Ht, sigma] of MIMO system
    alpha: Prior probabilities
    m: Modulation alphabet
    nudelta0m_iter: number of iterations
    binary: Select special binary case
    OUTPUT
    s: estimated symbols
    w_m: estimated prob of symbols, one-hot vectors
    '''
    def __init__(self, num_iter = 64, m = np.array([-1, 1]), alpha = np.array([0.5, 0.5]), binary = False, gamma0 = 1, delta0 = 1,
                    **kwargs):
        super(amp, self).__init__(**kwargs)
        self.it = tf.constant(num_iter)
        self.mt = m[tf.newaxis, tf.newaxis, ...]
        self.alpha = tf.constant(value = alpha)
        self.M = m.shape[0]
        gamma0 = tf.constant(gamma0, shape = (num_iter + 1), dtype = 'float32')
        delta0 = tf.constant(delta0, shape = (num_iter), dtype = 'float32')
        # Starting point
        self.s0 = tf.constant(tf.tensordot(alpha, m, axes = 1)[tf.newaxis, :]) # a-priori mean
        self.amp_layer = []
        self.gamma = []
        self.delta = []
        for ii in range(num_iter + 1):
            if binary == True:
                if ii == 0:
                    self.amp_layer.append(amp_bin_layer(m = self.mt, alpha = self.alpha, gamma = gamma0[ii], delta = delta0[ii], first_iter = True))
                elif ii == num_iter:
                    self.amp_layer.append(amp_bin_layer(m = self.mt, alpha = self.alpha, gamma = gamma0[ii], delta = 0, last_iter = True))
                else:
                    self.amp_layer.append(amp_bin_layer(m = self.mt, alpha = self.alpha, gamma = gamma0[ii], delta = delta0[ii]))
            else:
                if ii == 0:
                    self.amp_layer.append(amp_layer(m = self.mt, alpha = self.alpha, gamma = gamma0[ii], delta = delta0[ii], first_iter = True))
                elif ii == num_iter:
                    self.amp_layer.append(amp_layer(m = self.mt, alpha = self.alpha, gamma = gamma0[ii], delta = 0, last_iter = True))
                else:
                    self.amp_layer.append(amp_layer(m = self.mt, alpha = self.alpha, gamma = gamma0[ii], delta = delta0[ii]))
            self.gamma.append(self.amp_layer[-1].gamma)
            if not ii == num_iter:
                self.delta.append(self.amp_layer[-1].delta)

    @tf.function#(jit_compile = True)
    def call(self, inputs):
        [yt, Ht, sigmat0] = inputs
        N0 = sigmat0 ** 2
        beta = Ht.shape[-1] / Ht.shape[-2]
        tau = beta * tf.reduce_mean(self.alpha * self.mt ** 2) / N0
        HH = tf.matmul(Ht, Ht, transpose_a = True)
        yH = tf.squeeze(tf.matmul(tf.expand_dims(yt, axis = 1), Ht), axis = 1)
        rH = yH - tf.squeeze(tf.matmul(HH, tf.expand_dims(self.s0, axis = -1)), axis = -1)
        s = self.s0 * tf.ones_like(Ht[:, 0, :])
        ft = self.alpha
        for layer in self.amp_layer:
            s, rH, tau, ft = layer(s, rH, tau, HH, yH, N0, beta)
        # ft = w_m
        xt = s
        return ft, xt


class amp_layer(tf.keras.layers.Layer):
    """The AMP layer
    """
    def __init__(self, m = np.array([-1, 1]), alpha = np.array([0.5, 0.5]), gamma = 1, delta = 1, first_iter = False, last_iter = False):
        super().__init__()
        # TODO: m and alpha trainable? -> needs to be changed in constellation object
        # and given to layers/function call
        self.mt = tf.constant(m)
        self.alpha = tf.constant(value = alpha)
        self.first_iter = first_iter
        self.last_iter = last_iter
        self.gamma = self.add_weight(shape = (1,),
                             initializer = tf.keras.initializers.Constant(value = gamma),
                             trainable = True,
                             name = 'gamma')
        if not self.last_iter:
            self.delta = self.add_weight(shape = (1,),
                                initializer = tf.keras.initializers.Constant(value = delta),
                                trainable = True,
                                name = 'delta')

    @tf.function#(jit_compile = True)
    def call(self, s, rH, tau, HH, yH, N0, beta):
        # Start of new iteration
        tau = tau * self.gamma
        z = s + rH
        var_F = N0 * (1 + tau)
        arg = - 1 / 2 / var_F[:, tf.newaxis, tf.newaxis] * (z[..., tf.newaxis] - self.mt) ** 2 + tf.math.log(self.alpha[tf.newaxis, ...])
        w_m = tf.math.softmax(arg)
        s = tf.reduce_sum(w_m * self.mt, axis = -1)
        if not self.last_iter:
            G = tf.reduce_sum(w_m * (self.mt - s[..., tf.newaxis]) ** 2, axis = -1)
            tau_old = tau * self.delta
            tau = beta / N0 * tf.reduce_mean(G, axis = -1)
            rH = yH - self.delta * tf.squeeze(tf.matmul(HH, tf.expand_dims(s, axis = -1)), axis = -1) + (tau / (tau_old + 1))[:, tf.newaxis] * rH

        # taui_abs = tf.math.abs(self.taui) # no negative values for tau !
        # if self.first_iter == True:
        #     ft = tf.nn.softmax((tf.math.log(self.alpha) + G) * 1, axis = -1)
        # else:
        #     ft = tf.nn.softmax((tf.math.log(self.alpha) + G) * taui_abs, axis = -1)
        return s, rH, tau, w_m


class amp_bin_layer(tf.keras.layers.Layer):
    """The binary AMP layer
    """
    def __init__(self, m = np.array([-1, 1]), alpha = np.array([0.5, 0.5]), gamma = 1, delta = 1, first_iter = False, last_iter = False):
        super().__init__()
        # TODO: m and alpha trainable? -> needs to be changed in constellation object
        # and given to layers/function call
        self.mt = tf.constant(m)
        self.alpha = tf.constant(value = alpha)
        self.first_iter = first_iter
        self.last_iter = last_iter
        self.gamma = self.add_weight(shape = (1,),
                             initializer = tf.keras.initializers.Constant(value = gamma),
                             trainable = True,
                             name = 'gamma')
        if not self.last_iter:
            self.delta = self.add_weight(shape = (1,),
                                initializer = tf.keras.initializers.Constant(value = delta),
                                trainable = True,
                                name = 'delta')

    @tf.function#(jit_compile = True)
    def call(self, s, rH, tau, HH, yH, N0, beta):
        # Start of new iteration
        tau = tau * self.gamma
        z = s + rH
        var_F = N0 * (1 + tau)
        s = tf.math.tanh(z / var_F[:, tf.newaxis])
        if not self.last_iter:
            tau_old = tau
            tau = beta / N0 * tf.reduce_mean(1 - s ** 2, axis = -1)
            rH = yH - self.delta * tf.squeeze(tf.matmul(HH, tf.expand_dims(s, axis = -1)), axis = -1) + (tau / (tau_old + 1))[:, tf.newaxis] * rH
        
        w_m = tf.concat([(1 + tf.math.sign(self.mt[..., 0]) * s[..., tf.newaxis]) / 2, (1 + tf.math.sign(self.mt[..., 1]) * s[..., tf.newaxis]) / 2], axis = -1)
        return s, rH, tau, w_m



def conventional_training(model, num_training_iterations, training_batch_size, ebno_db_train, it_print = 100):
    # Optimizer used to apply gradients
    # try also different optimizers or different hyperparameters
    optimizer = tf.keras.optimizers.Adam() # learning_rate = 1e-2)
    # clip_value_grad = 10 # gradient clipping for stable training convergence
    # bmi is used as metric to evaluate the intermediate results
    bmi = BitwiseMutualInformation() # nur zur Auswertung
    

    # First iteration loss
    start_time = time.time()
    start_time2 = time.time()
    b, b_hat, loss = model(training_batch_size, ebno_db_train[0], ebno_db_train[1])
    ber = compute_ber(b, b_hat)
    mi = bmi(b, b_hat).numpy() # calculate bit-wise mutual information
    l = loss.numpy() # copy loss to numpy for printing
    print(f"It: {0}/{num_training_iterations}, Train loss: {l:.6f}, BER: {ber:.4f}, BMI: {mi:.3f}, Time: {time.time() - start_time2:04.2f}s, Tot. time: ".format() + print_time(time.time() - start_time))
    bmi.reset_states() # reset the BMI metric
    start_time2 = time.time()
    
    for it in range(num_training_iterations):
        # Forward pass
        with tf.GradientTape() as tape:
            b, b_hat, loss = model(training_batch_size, ebno_db_train[0], ebno_db_train[1])
        # Computing and applying gradients
        grads = tape.gradient(loss, model.trainable_weights)
        # grads = tf.clip_by_value(grads, -clip_value_grad, clip_value_grad, name=None)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # Printing periodically the training metrics
        if (it + 1) % it_print == 0: # evaluate every it_print iterations
            ber = compute_ber(b, b_hat)
            mi = bmi(b, b_hat).numpy() # calculate bit-wise mutual information
            l = loss.numpy() # copy loss to numpy for printing
            print(f"It: {it + 1}/{num_training_iterations}, Train loss: {l:.6f}, BER: {ber:.4f}, BMI: {mi:.3f}, Time: {time.time() - start_time2:04.2f}s, Tot. time: ".format() + print_time(time.time() - start_time))
            bmi.reset_states() # reset the BMI metric
            start_time2 = time.time()


def script1():
    '''All CMDNet curves from the journal article for dimension 32x32 (effective dimension: 64x64)
    '''
    ber_plot = PlotBER()
    code = 0
    trainbit = 0
    mod = 'qam'
    num_tx_ant = 32
    num_rx_ant = 32
    Nit = 2 * num_tx_ant    # 64
    num_bits_per_symbol = 2
    constellation = Constellation(mod, num_bits_per_symbol, trainable = False)
    # Test params
    snr_range = np.arange(1, 18, 1) # [1, 13, 1], [1, 31, 1], [-3, 16, 0.5]
    # snr_range = np.arange(1, 12, 1), # -3, 16, 0.5
    batch_size = 10020 # int(10000 / (model1.num_tx_ant * model1.num_tx_ant / 4)), # 4096
    max_mc_iter = 10 # 1000
    num_target_block_errors = 100
    
    # Load starting point
    algo0 = algo_mmse(constellation)
    algo1 = algo_amp(Nit, constellation, num_tx_ant)
    algo2 = algo_amp(Nit, constellation, num_tx_ant, binary = True)
    
    ## Old weight loading
    # saveobj2 = mf.savemodule('npz')
    # train_hist2 = mf.training_history()
    # sim_set = {'Mod': 'QPSK', 'Nr': 64, 'Nt': 64, 'L': 64,} # BPSK, QPSK, QAM16
    # fn = mf.filename_module('trainhist_', 'curves', 'CMD', '_binary_tau0.1', sim_set) # _binary_tau0.1, _tau0.1, _convex, _binary_tau0.075, '_binary_splin'
    # train_hist2.dict2obj(saveobj2.load(fn.pathfile))
    # [delta0, taui0] = train_hist2.params[-1]
    
    algo3 = algo_cmdnet(Nit, constellation, num_tx_ant, binary = True) #, taui0 = taui0, delta0 = delta0)
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 * num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit,}
    fn = filename_module('data_MIMO_sionna', 'weights_', algo3.algo_name, 'binary_tau0.1', sim_set)
    algo3.load_weights(fn.pathfile)
    
    algo4 = algo_cmdnet(Nit, constellation, num_tx_ant, binary = False) #, taui0 = taui0, delta0 = delta0)
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 * num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit,}
    fn = filename_module('data_MIMO_sionna', 'weights_', algo4.algo_name, 'tau0.1', sim_set)
    algo4.load_weights(fn.pathfile)
    
    algo5 = algo_cmdnet(16, constellation, num_tx_ant, binary = True) #, taui0 = taui0, delta0 = delta0
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 * num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': 16,}
    fn = filename_module('data_MIMO_sionna', 'weights_', algo5.algo_name, 'binary_tau0.075', sim_set)
    algo5.load_weights(fn.pathfile)
    
    algo6 = algo_cmdnet(Nit, constellation, num_tx_ant, binary = True) #, taui0 = taui0, delta0 = delta0
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 * num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit,}
    fn = filename_module('data_MIMO_sionna', 'weights_', algo6.algo_name, 'binary_splin', sim_set)
    algo6.load_weights(fn.pathfile)
    
    delta0, taui0 = CMD_initpar(M = 2, L = 64, typ = 'default', min_val = 0.1)
    algo7 = algo_cmdnet(Nit, constellation, num_tx_ant, binary = True, taui0 = taui0, delta0 = delta0)
    delta0, taui0 = CMD_initpar(M = 2, L = 64, typ = 'linear', min_val = 0.01)
    algo8 = algo_cmdnet(Nit, constellation, num_tx_ant, binary = True, taui0 = taui0, delta0 = delta0)

    # model1.algo.load_weights('MIMO_sionna/weights')
    model0 = Model(algo = algo0, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)
    model1 = Model(algo = algo1, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)
    model2 = Model(algo = algo2, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)
    model3 = Model(algo = algo3, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)
    model4 = Model(algo = algo4, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)
    model5 = Model(algo = algo5, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)
    model6 = Model(algo = algo6, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)
    model7 = Model(algo = algo7, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)
    model8 = Model(algo = algo8, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)


    ber_plot.simulate(model0,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = "LMMSE (Uncorrelated)",
            show_fig = False);
    ber_plot.simulate(model1,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = "AMP N_it = 64",
            show_fig = False);
    ber_plot.simulate(model2,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = "AMP bin N_it = 64",
            show_fig = False);
    ber_plot.simulate(model3,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = 'CMDNet bin N_L = 64',
            show_fig = False);
    ber_plot.simulate(model4,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = 'CMDNet N_L = 64',
            show_fig = False);
    ber_plot.simulate(model5,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = 'CMDNet N_L = 16',
            show_fig = False);
    ber_plot.simulate(model6,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = 'CMDNet splin',
            show_fig = False);
    ber_plot.simulate(model7,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = 'CMDNet spdef Ntrain=0',
            show_fig = False);
    ber_plot.simulate(model8,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = 'CMDNet splin Ntrain=0',
            # save_fig = True,
            show_fig = True);

    tplt.save('plots/MIMO_sionna_test.tikz')


def script2():
    '''All CMDNet curves from the journal article for dimension 8x8 (effective dimension: 16x16)
    '''
    ber_plot = PlotBER()
    code = 0
    trainbit = 0
    mod = 'qam'
    num_tx_ant = 8
    num_rx_ant = 8
    Nit = 2 * num_tx_ant    # 16
    num_bits_per_symbol = 2
    constellation = Constellation(mod, num_bits_per_symbol, trainable = False)
    # Test params
    snr_range = np.arange(1, 16, 1) # [1, 13, 1], [1, 31, 1], [-3, 16, 0.5]
    # snr_range = np.arange(1, 12, 1), # -3, 16, 0.5
    batch_size = 10020 # int(10000 / (model1.num_tx_ant * model1.num_tx_ant / 4)), # 4096
    max_mc_iter = 100 # 1000
    num_target_block_errors = 100
    
    # Load starting point
    algo0 = algo_mmse(constellation)
    # algo1 = algo_amp(Nit, constellation, num_tx_ant)
    algo2 = algo_amp(Nit, constellation, num_tx_ant, binary = True)
    
    algo3 = algo_cmdnet(Nit, constellation, num_tx_ant, binary = True) #, taui0 = taui0, delta0 = delta0)
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 * num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit,}
    fn = filename_module('data_MIMO_sionna', 'weights_', algo3.algo_name, 'binary', sim_set)
    algo3.load_weights(fn.pathfile)

    model0 = Model(algo = algo0, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)
    # model1 = Model(algo = algo1, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)
    model2 = Model(algo = algo2, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)
    model3 = Model(algo = algo3, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)


    ber_plot.simulate(model0,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = "LMMSE (Uncorrelated)",
            show_fig = False);
    # ber_plot.simulate(model1,
    #         snr_range,
    #         batch_size = batch_size,
    #         max_mc_iter = max_mc_iter,
    #         num_target_block_errors = num_target_block_errors,
    #         legend = "AMP N_it = 16",
    #         show_fig = False);
    ber_plot.simulate(model2,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = "AMP bin N_it = 16",
            show_fig = False);
    ber_plot.simulate(model3,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = 'CMDNet bin N_L = 16',
            # save_fig = True,
            show_fig = True);

    tplt.save('plots/MIMO_sionna_test.tikz')


def script3():
    '''All CMDNet curves from the journal article for QAM16 modulation dimension 32x32 (effective 4-ASK modulation and dimension: 64x64)
    '''
    ber_plot = PlotBER()
    code = 0
    trainbit = 0
    mod = 'qam'
    num_tx_ant = 32
    num_rx_ant = 32
    Nit = 2 * num_tx_ant    # 64
    num_bits_per_symbol = 4
    constellation = Constellation(mod, num_bits_per_symbol, trainable = False)
    # Test params
    snr_range = np.arange(6, 31, 1)
    batch_size = 10020
    max_mc_iter = 100 # 1000
    num_target_block_errors = 1000
    
    # Load starting point
    algo0 = algo_mmse(constellation)
    algo1 = algo_amp(Nit, constellation, num_tx_ant)
    
    algo2 = algo_cmdnet(Nit, constellation, num_tx_ant) #, taui0 = taui0, delta0 = delta0)
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 * num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit,}
    fn = filename_module('data_MIMO_sionna', 'weights_', algo2.algo_name, 'convex', sim_set)
    algo2.load_weights(fn.pathfile)

    model0 = Model(algo = algo0, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)
    model1 = Model(algo = algo1, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)
    model2 = Model(algo = algo2, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)


    ber_plot.simulate(model0,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = "LMMSE (Uncorrelated)",
            show_fig = False);
    ber_plot.simulate(model1,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = "AMP bin N_it = 64",
            show_fig = False);
    ber_plot.simulate(model2,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = 'CMDNet bin N_L = 64',
            # save_fig = True,
            show_fig = True);

    tplt.save('plots/MIMO_sionna_test.tikz')


def script4():
    '''All CMDNet curves with code from the journal article for QPSK modulation and dimension 32x32 (effective BPSK modulation and dimension: 64x64)
    TODO: Somehow CMDNet with code does not work here...
    Idea: Try using with old BP implementation -> no change -> problem lies within new cmdnet implementation
    '''
    ber_plot = PlotBER()
    mod = 'qam'
    num_tx_ant = 32
    num_rx_ant = 32
    Nit = 2 * num_tx_ant    # 64
    num_bits_per_symbol = 2
    constellation = Constellation(mod, num_bits_per_symbol, trainable = False)
    n = 128     # 128, 1024
    k = 64      # 64, 512
    Ncit = 10
    # Test params
    snr_range = np.arange(3, 14, 1)     # np.arange(3, 19, 1)
    batch_size = 10
    max_mc_iter = 10 # 100
    num_target_block_errors = 100
    
    # Load starting point
    algo0 = algo_mmse(constellation)
    algo1 = algo_amp(Nit, constellation, num_tx_ant)
    
    algo2 = algo_cmdnet(Nit, constellation, num_tx_ant) #, taui0 = taui0, delta0 = delta0)
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 * num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit,}
    fn = filename_module('data_MIMO_sionna', 'weights_', algo2.algo_name, 'binary_tau0.1', sim_set)
    algo2.load_weights(fn.pathfile)

    model0 = Model(algo = algo0, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation,
                    code = True, n = n, k = k, code_it = Ncit)
    model1 = Model(algo = algo1, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation,
                    code = True, n = n, k = k, code_it = Ncit)
    model2 = Model(algo = algo2, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation,
                    code = True, n = n, k = k, code_it = Ncit)
    model3 = Model(algo = algo2, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation)


    ber_plot.simulate(model0,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = "LMMSE (Uncorrelated)",
            add_ber = False,
            add_bler = True,
            show_fig = False);
    ber_plot.simulate(model1,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = "AMP bin N_it = 64",
            add_ber = False,
            add_bler = True,
            show_fig = False);
    ber_plot.simulate(model2,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = 'CMDNet bin N_L = 64',
            add_ber = False,
            add_bler = True,
            show_fig = False);
    ber_plot.simulate(model3,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = 'CMDNet bin N_L = 64 uncoded',
            add_ber = False,
            add_bler = True,
            # save_fig = True,
            show_fig = True);

    tplt.save('plots/MIMO_sionna_test.tikz')


def CMD_initpar(M, L, typ, k = 0, min_val = 0.1, tau_max_scale = 1):
    '''Function calculating a good starting point for CMD / loading starting point
    M: Number of classes / Modulation order
    L: Number of layers / iterations
    'typ': linear/constant
    k: parallel branches of CMDNet -> repetition of parameters
    min_val: minimum value of tau and delta
    '''
    tau_max = 1 / (M - 1) * tau_max_scale # for 4ASK: tau_max_scale = 2
    delta_max = 1
    tau_min = min_val # * tau_max # default: 0.1
    delta_min = min_val # * delta_max # default: 0.1
    if  typ.casefold() == 'linear':
        # Linear decrease
        tau0 =  tau_max - (tau_max - tau_min) / L * np.linspace(0, L, L + 1)
        delta0 = delta_max - (delta_max - delta_min) / L * np.linspace(0, L, L)
        taui0 = 1 / tau0
    elif typ.casefold() == 'const':
        # Constant
        tau0 = tau_max * np.ones(L + 1)
        delta0 = delta_max  * np.ones(L)
        taui0 = 1 / tau0
    else:
        # Default: only linear decrease in taui
        tau0 =  tau_max - (tau_max - tau_min) / L * np.linspace(0, L, L + 1)
        delta0 = delta_max  * np.ones(L)
        taui0 = 1 / tau0
    
    if k != 0:
        tau0 = tau0[:, np.newaxis].repeat((k), axis = -1)
        delta0 = delta0[:, np.newaxis].repeat((k), axis = -1)
        taui0 = 1 / tau0

    return delta0, taui0


class filename_module():
    '''Class responsible for file names
    '''
    # Class Attribute
    name = 'File name creator'
    # Initializer / Instance Attributes
    def __init__(self, path, typename, algo, fn_ext, sim_set, code_set = 0):
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
        # Initialize
        self.generate_pathfile_MIMO()
    # Instance methods
    def generate_filename_MIMO(self):
        '''Generates file name
        '''
        if self.code:
            self.filename = self.typename + self.algoname + '_' + self.mod + '_{}_{}_{}_'.format(self.Nt, self.Nr , self.L) + self.code['code'] + self.code['dec'] + self.code['arch'] + '_' + self.fn_ext
        else:
            self.filename = self.typename + self.algoname + '_' + self.mod + '_{}_{}_{}'.format(self.Nt, self.Nr , self.L) + '_' + self.fn_ext
        return self.filename
    def generate_path_MIMO(self):
        '''Generates path name
        '''
        self.path = os.path.join(self.ospath, self.mod, '{}x{}'.format(self.Nt, self.Nr), self.fn_ext) # '/', '\\'
        return self.path
    def generate_pathfile_MIMO(self):
        '''Generates full path and filename
        '''
        self.generate_path_MIMO()
        self.generate_filename_MIMO()
        self.pathfile = os.path.join(self.path, self.filename)
        return self.pathfile


def script5():
    '''Exemplary training for binary CMDNet for QPSK modulation and dimension 32x32 (effective BPSK modulation and dimension: 64x64)
    TODO: Debug training with NaNs, worked in tensorflow 1...
    Training runs in eager mode, but not in graph mode. Why?
    '''
    tf.config.run_functions_eagerly(True)
    
    # Training
    code = 0
    trainbit = 0
    mod = 'qam'
    num_tx_ant = 32
    num_rx_ant = 32
    Nit = num_tx_ant * 2
    num_bits_per_symbol = 2
    constellation = Constellation(mod, num_bits_per_symbol, trainable = False)
    # Test params
    ber_plot = PlotBER()
    snr_range = np.arange(1, 18, 1)
    batch_size = 10020
    max_mc_iter = 10
    num_target_block_errors = 100
    
    # Algorithm and Model
    delta0, taui0 = CMD_initpar(M = 2, L = 64, typ = 'default', min_val = 0.1)
    algo1 = algo_cmdnet(Nit, constellation, num_tx_ant, binary = True, taui0 = taui0, delta0 = delta0)
    ## Load pretrained weights
    # sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 * num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit,}
    # fn = filename_module('data_MIMO_sionna', 'weights_', algo1.algo_name, 'binary_tau0.1', sim_set)
    # algo1.load_weights(fn.pathfile)
    model1 = Model(algo = algo1, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)
    
    # Training parameters
    batch_size = 500    # w/o code: 1000 -> 500 ?, w code: 10/1
    train_iter = 1000
    ebno_db_train = [7, 26] # w/o code: [7, 26], w code: [0, 3] # QAM16: [10, 33]
    conventional_training(model1, train_iter, batch_size, ebno_db_train, it_print = 100)
    
    # Save after training
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 * num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit,}
    fn = filename_module('data_MIMO_sionna', 'weights_', algo1.algo_name, 'test', sim_set)
    # Path has to be shorter...
    path = os.path.join(fn.path, 'weights')
    model1.algo.save_weights(path) # fn.pathfile
    
    ## Debug: Forward pass
    # model1 = Model(sel_algo = 1, mod = mod, num_bits_per_symbol = 2, code = code, trainbit = trainbit)
    # with tf.GradientTape() as tape:
    #     b, b_hat, loss = model1(batch_size, ebno_db_train[0], ebno_db_train[1])
    # # Computing and applying gradients
    # grads = tape.gradient(loss, model1.trainable_weights)

    # Comparison
    algo2 = algo_cmdnet(Nit, constellation, num_tx_ant, binary = True, taui0 = taui0, delta0 = delta0)
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 * num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit,}
    fn = filename_module('data_MIMO_sionna', 'weights_', algo1.algo_name, 'binary_tau0.1', sim_set)
    algo2.load_weights(fn.pathfile)
    model2 = Model(algo = algo2, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)

    ber_plot.simulate(model1,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = 'CMDNet trained',
            show_fig = False);
    ber_plot.simulate(model2,
            snr_range,
            batch_size = batch_size,
            max_mc_iter = max_mc_iter,
            num_target_block_errors = num_target_block_errors,
            legend = 'CMDNet bin N_L = 64',
            show_fig = True);





if __name__ == '__main__':
#     my_func_main()
# def my_func_main():
    # tf.debugging.enable_check_numerics()
    # tf.config.run_functions_eagerly(True)
    # tf.keras.backend.set_floatx('float64')
    
    # Choose example script
    example = 0 # -1: Debug, 0: CMDNet QPSK 64x64, 1: CMDNet QPSK 16x16, 2: CMDNet QAM16 64x64, 3: CMDNet with code (not working...), 4: Training of CMDNet (not working...)
    
    
    if example == 0:
        script1()
    elif example == 1:
        script2()
    elif example == 2:
        script3()
    elif example == 3:
        script4()
    elif example == 4:
        script5()
    else:
        # Debugging
        code = 0
        trainbit = 0
        mod = 'qam'
        binary = True
        num_tx_ant = 32
        num_rx_ant = 32
        Nit = 64
        num_bits_per_symbol = 2
        constellation = Constellation(mod, num_bits_per_symbol, trainable = False)
        
        algo1 = algo_cmdnet(Nit, constellation, num_tx_ant, binary = binary) #, taui0 = taui0, delta0 = delta0)
        sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 * num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit,}
        fn = filename_module('data_MIMO_sionna', 'weights_', algo1.algo_name, 'binary_tau0.1', sim_set)
        algo1.load_weights(fn.pathfile)
        # algo1 = algo_amp(Nit, constellation, num_tx_ant)
        # algo1 = algo_mmse(constellation)
        model1 = Model(algo = algo1, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit,
                        n = 128, k = 64, code_it = 10, code_train = False)


        # DEBUGGEN: Erstmal cmdnet richtig programmieren!!!
        dtype0 = 'float32'
        batch_size = 100 # int(10000 / (model1.num_tx_ant * model1.num_tx_ant / 4)) # 4096
        if model1.code == True:
            b = model1.binary_source([batch_size, model1.num_tx_ant, model1.k])
            c = model1.encoder(b)
            ## Own BP implementation
            # c = mf.encoder(b, model1.G)
        else:
            b = model1.binary_source([batch_size, model1.num_tx_ant])
            c = b
        shapec = tf.shape(c)
        x = model1.mapper(c)
        shape = tf.shape(x)
        x = tf.reshape(x, [-1, model1.num_tx_ant])
        ebn0_debug = 6 # 12
        ebno_db = tf.random.uniform(shape = [x.shape[0]], minval = ebn0_debug, maxval = ebn0_debug)
        no = 1 / 2 * ebnodb2no(ebno_db, model1.num_bits_per_symbol, 1) # model1.coderate)
        no_scale = model1.num_rx_ant * 2 / 2  # scaling of noise variance necessary since channel has Rayleigh taps with variance 1
        # no *= np.sqrt(model1.num_rx_ant)    # weil die rayleigh taps von h jeweils power 1 haben...
        y_scale = np.sqrt(2 / model1.num_rx_ant)
        if not model1.mod == 'pam':
            H_scale = np.sqrt(2 / (2 * model1.num_rx_ant))
        sigmat0 = tf.math.sqrt(no)
        y, h = model1.channel([x, no * no_scale])

        # # MMSE debug
        # ## x_hat2, Phi_ee = mf.mmse([tf.math.real(y) * H_scale, tf.math.real(h) * H_scale, tf.repeat(sigmat0, y.shape[0])])
        # ## no_eff2 = mf.tdiag2vec(Phi_ee)
        # if model1.mod == 'pam':
        #     Ht = y_scale * tf.cast(tf.math.real(h), dtype = 'complex64')
        #     yt = y_scale * tf.cast(tf.math.real(y), dtype = 'complex64')
        #     s = tf.complex(no[..., tf.newaxis, tf.newaxis] * tf.eye(model1.num_rx_ant, model1.num_rx_ant), 0.0)
        # else:
        #     Ht = H_scale * tf.concat([tf.concat([tf.math.real(h), -tf.math.imag(h)], axis = -1), tf.concat([tf.math.imag(h), tf.math.real(h)], axis = -1)], axis = 1)
        #     yt = y_scale * tf.concat([tf.math.real(y), tf.math.imag(y)], axis = -1)
        #     s = tf.complex(no[..., tf.newaxis, tf.newaxis] * tf.eye(2 * model1.num_rx_ant, 2 * model1.num_rx_ant), 0.0)
        #     # Ht = H_scale * h
        #     # yt = y_scale * y
        #     # s = tf.complex(tf.expand_dims(tf.expand_dims(no, axis = -1), axis = -1) * tf.eye(self.num_rx_ant, self.num_rx_ant), 0.0)
        # # x_hat, no_eff = lmmse_equalizer(y * H_scale, h * H_scale, s)
        # x_hat, no_eff = lmmse_equalizer(tf.cast(yt, dtype = 'complex64'), tf.cast(Ht, dtype = 'complex64'), s)
        # if model1.mod == 'pam':
        #     x_hat = tf.reshape(x_hat, shape)
        #     no_eff = tf.reshape(no_eff, shape)
        # else:
        #     x_hat = tf.reshape(x_hat[:, :model1.num_tx_ant] + 1j * x_hat[:, model1.num_tx_ant:], shape)
        #     no_eff = tf.reshape(no_eff[:, :model1.num_tx_ant] + no_eff[:, model1.num_tx_ant:], shape)
        
        # llr_c = model1.algo.demapper([tf.cast(x_hat, dtype = 'complex64'), tf.cast(no_eff, dtype = 'float32')])
        # # # b_hat = model1.decoder(llr_c)
        # # ft, x_hat, llr_c, p_c = model1.algo([y, h, sigmat0])

        # Wrapper for CMDNet
        m_mapper = constellation.points
        if mod == 'pam':
            M = 2 ** num_bits_per_symbol
            m = tf.math.real(m_mapper) # self.mapper.constellation.points
            alpha = 1 / M * np.ones((num_tx_ant, M), dtype = 'float32')
        else:
            M = 2 ** int(num_bits_per_symbol / 2)
            c_poss = tf_int2bin(np.array(range(0, 2 ** int(num_bits_per_symbol / 2))), int(num_bits_per_symbol / 2))
            b_power = 2 ** np.array(range(0, num_bits_per_symbol))[::-1][::2]
            ind_m = np.sum(b_power * c_poss, axis = -1)
            m = tf.math.real(tf.gather(m_mapper, ind_m, axis = -1)) * np.sqrt(2)
            alpha = 1 / M * np.ones((2 * num_tx_ant, M), dtype = 'float32')
        y_scale = np.sqrt(2 / num_rx_ant)
        # Wrapper
        if mod == 'pam':
            Ht = y_scale * tf.math.real(h)
            yt = y_scale * tf.math.real(y)
        else:
            H_scale = np.sqrt(2 / (2 * num_rx_ant))
            Ht = H_scale * tf.concat([tf.concat([tf.math.real(h), -tf.math.imag(h)], axis = -1), tf.concat([tf.math.imag(h), tf.math.real(h)], axis = -1)], axis = 1)
            yt = y_scale * tf.concat([tf.math.real(y), tf.math.imag(y)], axis = -1)
        # Equalization
        # cmdnet0 = cmdnet(64, m, alpha)
        # ft, xt = cmdnet0([yt, Ht, sigmat0])
        # amp0 = amp(64, m, alpha, binary = False)
        # ft, xt = amp0([yt, Ht, sigmat0])
        # ft, xt = model1.algo.amp([yt, Ht, sigmat0])
        ft, xt, llr_c, p_c_hat = model1.algo([y, h, sigmat0])
        # llr_c, p_c_hat = model1.algo.symprob2llr(ft)
        # ft, x_hat, llr_c, p_c = model1.algo([y, h, sigmat0])
        c_cl = tf.reshape(c, [-1, model1.num_tx_ant, model1.num_bits_per_symbol])
        if model1.mod == 'pam':
            M = 2 ** model1.num_bits_per_symbol
            cl = tf_bin2int(c_cl, axis = -1)
        else:
            M = 2 ** int(model1.num_bits_per_symbol / 2)
            # e.g., c = 101010 -> c_re = 111, c_im = 000
            c_cl_re = tf_bin2int(c_cl[..., ::2], axis = -1)
            c_cl_im = tf_bin2int(c_cl[..., 1::2], axis = -1)
            # concatenate classes of real and imaginary part as in equivalent real-valued system model
            cl = tf.concat([c_cl_re, c_cl_im], axis = -1)
        z = tf.one_hot(cl, depth = M)
        loss = tf.reduce_mean(tf.reduce_mean(tf.keras.losses.categorical_crossentropy(z, ft, axis = -1), axis = -1))

        # Get symbol indices for the transmitted symbols
        # if not model1.mod == 'pam':
        #     # llr_c = llr_c[..., tf.newaxis]
        #     # p_c = p_c[..., tf.newaxis]
        #     llr_c = tf.concat([llr_c[:, :model1.num_tx_ant], llr_c[:, model1.num_tx_ant:]], axis = -1)
        #     llr_c = tf.concat([llr_c[..., 0::2], llr_c[..., 1::2]], axis = -1)
        #     p_c_hat = tf.concat([p_c_hat[:, :model1.num_tx_ant], p_c_hat[:, model1.num_tx_ant:]], axis = -1)
        #     p_c = tf.concat([p_c_hat[..., 0::2], p_c_hat[..., 1::2]], axis = -1)
        llr_c = tf.reshape(llr_c, shapec)
        if model1.code == True:
            llr_hat = model1.decoder(llr_c)
            ## Own bp implementation
            # shape_llr = llr_c.shape
            # llr_c = tf.reshape(llr_c, [-1, 128])
            # llr_hat, _, _ = mf.bp_decoder(llr_c, model1.H, 10, 0)
            # llr_hat = tf.reshape(llr_hat[..., 0:64], [shape_llr[0], shape_llr[1], -1])
        else:
            llr_hat = llr_c
        p_b_hat = model1.algo.symprob2llr.llr2prob(llr_hat)
        b_hat = tf.cast(llr_hat > 0, dtype = 'float32')
        bce = tf.keras.losses.BinaryCrossentropy()
        loss = bce(b, p_b_hat)


