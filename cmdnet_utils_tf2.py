#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 01 14:33:08 2022

CMDNet (Concrete MAP Detection Network) for MIMO - Keras layers

@author: beck
"""

# LOADED PACKAGES
# Tensorflow packages
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import numpy as np
import tensorflow as tf
# Import Sionna
from sionna.mapping import Demapper
from sionna.mimo import lmmse_equalizer

# Sionna only works with float32 precision in version 0.9.0
GLOBAL_PRECISION = 'float32'


@tf.function  # (jit_compile = True)
def tf_bin2int(b, axis=-1, dtype='int64'):
    '''Tensorflow version: Convert a N-bit bit vector [b] across dimension [axis] into positive integer num [cl]
    Maximum integer number with uint64 is 9223372036854775807
    '''
    b_power = 2 ** tf.range(tf.shape(b)[axis], dtype=b.dtype)[::-1]
    cl = tf.cast(tf.reduce_sum(b * b_power, axis=axis), dtype=dtype)
    return cl


@tf.function  # (jit_compile = True)
def tf_int2bin(x, N):
    '''Tensorflow version: Convert a positive integer num into an N-bit bit vector
    Limited up to N = 64 bits and 2 ** 64 numbers (!!!)
    '''
    return tf.math.mod(tf.bitwise.right_shift(tf.expand_dims(x, 1), tf.range(N, dtype=x.dtype)), 2)


class TFSymprob2LLR():
    '''Convert symbol probabilities to llrs and probabilities of each transmitted bit
    Note: Gray Mapping assumed for bit to symbol mapping
    '''
    # Class Attribute
    name = 'Tensorflow version of symprob2llr'
    # Initializer / Instance Attributes

    def __init__(self, M, b=0):
        '''
        M: modulation order
        b: bit reference, LLR = p(b = 0) / p(b = 1) vs. LLR = p(b = 1) / p(b = 0)
        '''
        self.M = M
        self.c_poss = tf_int2bin(np.array(range(0, M)), int(np.log2(M)))
        self.mask = tf.cast((self.c_poss == b)[
            np.newaxis, np.newaxis, ...], dtype=GLOBAL_PRECISION)
    # Instance methods

    # @tf.function  # (jit_compile = True)
    def __call__(self, p_m):
        '''
        p_m: symbol probability
        llr_c: log-likelihood ratio of bit c
        p_c: bit probability of c being b = 0 or b = 1 according to bit reference
        '''
        eps = tf.constant(1e-20, dtype=p_m.dtype)
        p_c = tf.reduce_sum(tf.expand_dims(p_m, axis=-1) * self.mask, axis=-2)
        # avoid NaN, -infinity with term above
        llr_c = tf.math.log((p_c + eps) / (1 - p_c + eps))
        llr_c = tf.clip_by_value(llr_c, -1e9, 1e9)   # avoid infinity
        return llr_c, p_c

    # @tf.function  # (jit_compile = True)
    def llr2prob(self, llr):
        '''
        Convert bit (!) llr into bit probability
        llr: log-likelihood ratio
        p_c: bit probability
        '''
        llr = tf.clip_by_value(
            llr, -90, 20)   # avoid NaN, for llr -> +infty: infty/infty = NaN
        p_c = tf.math.exp(llr) / (tf.math.exp(llr) + 1)
        return p_c


class AlgoMMSE(tf.keras.Model):
    """The MMSE algorithm with llr output
    """

    def __init__(self, constellation,
                 **kwargs):
        super(AlgoMMSE, self).__init__(**kwargs)
        self.algo_name = 'MMSE'
        self.demapper = Demapper('app', constellation=constellation)
        self.mod = constellation._constellation_type
        if self.mod == 'pam':
            self.M = constellation.points.shape[0]
        else:
            self.M = int(np.log2(constellation.points.shape[0]))
        self.symprob2llr = TFSymprob2LLR(self.M, b=1)

    @tf.function  # (jit_compile = True)
    def call(self, inputs, complex_model=2):
        [y, h, sigmat0] = inputs
        no = sigmat0 ** 2
        num_rx_ant = h.shape[-2]
        num_tx_ant = h.shape[-1]
        y_scale = np.sqrt(2 / num_rx_ant)
        if self.mod == 'pam':
            complex_model = 0
            Ht = y_scale * tf.cast(tf.math.real(h), dtype='complex64')
            yt = y_scale * tf.cast(tf.math.real(y), dtype='complex64')
            s = tf.complex(no[..., tf.newaxis, tf.newaxis] *
                           tf.eye(num_rx_ant, num_rx_ant), 0.0)
        else:
            if complex_model == 1:
                # Complex: Version 1
                Ht = np.sqrt(1 / num_rx_ant) * h
                yt = np.sqrt(1 / num_rx_ant) * y
                s = tf.complex(no[..., tf.newaxis, tf.newaxis]
                               * tf.eye(num_rx_ant, num_rx_ant), 0.0)
            elif complex_model == 2:
                # Complex: Version 2 without scaling
                no2 = no * num_rx_ant
                Ht = h
                yt = y
                s = tf.complex(no2[..., tf.newaxis, tf.newaxis]
                               * tf.eye(num_rx_ant, num_rx_ant), 0.0)
            else:
                # Real-valued
                # H_scale = np.sqrt(2 / (2 * num_rx_ant)) # Not used in combination with demapper
                Ht = y_scale * tf.concat([tf.concat([tf.math.real(h), -tf.math.imag(
                    h)], axis=-1), tf.concat([tf.math.imag(h), tf.math.real(h)], axis=-1)], axis=1)
                yt = y_scale * \
                    tf.concat([tf.math.real(y), tf.math.imag(y)], axis=-1)
                s = tf.complex(no[..., tf.newaxis, tf.newaxis]
                               * tf.eye(2 * num_rx_ant, 2 * num_rx_ant), 0.0)

        x_hat, no_eff = lmmse_equalizer(
            tf.cast(yt, dtype='complex64'), tf.cast(Ht, dtype='complex64'), s)
        # demapper expects complex x_hat normalized w.r.t. Re and Im part (!!!)
        if self.mod == 'qam' and complex_model == 0:
            x_hat = x_hat[:, :num_tx_ant] + 1j * x_hat[:, num_tx_ant:]
            no_eff = no_eff[:, :num_tx_ant] + no_eff[:, num_tx_ant:]
        # x_hat = tf.reshape(x_hat, shape)
        # no_eff = tf.reshape(no_eff, shape)

        llr_c = self.demapper(
            [tf.cast(x_hat, dtype='complex64'), tf.cast(no_eff, dtype=GLOBAL_PRECISION)])
        p_c = self.symprob2llr.llr2prob(llr_c)
        ft = []
        if not self.mod == 'pam':
            llr_c = tf.expand_dims(llr_c, axis=-1)
            p_c = tf.expand_dims(p_c, axis=-1)
            llr_c = tf.reshape(
                llr_c, [-1, int(llr_c.shape[1] / self.M), self.M])
            # llr_c = tf.concat([llr_c[..., 0::2], llr_c[..., 1::2]], axis=-1)
            p_c = tf.reshape(
                p_c, [-1, int(p_c.shape[1] / self.M), self.M])
            # p_c = tf.concat([p_c[..., 0::2], p_c[..., 1::2]], axis=-1)
        return ft, x_hat, llr_c, p_c
