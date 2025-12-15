#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 01 14:33:08 2022

Approximate Message Passing for MIMO - Keras layers

@author: beck
"""

# LOADED PACKAGES
import numpy as np
import tensorflow as tf
# Import Sionna
from sionna.mapping import Constellation
import cmdnet_utils_tf2 as cmd_utils_tf2

# Sionna only works with float32 precision in version 0.9.0
GLOBAL_PRECISION = 'float32'


class AlgoAMP(tf.keras.Model):
    """The AMP algorithm with llr output
    """

    def __init__(self, num_iter=64, const=Constellation('pam', 1), num_tx_ant=1, binary=False, gamma0=1, delta0=1,
                 **kwargs):
        super(AlgoAMP, self).__init__(**kwargs)
        self.algo_name = 'AMP'
        num_bits_per_symbol = const._num_bits_per_symbol
        self.mod = const._constellation_type
        m_mapper = const.points
        self.binary = binary
        M = m_mapper.shape[0]
        num_bits_per_symbol = int(np.log2(M))

        if self.mod == 'pam':
            M = 2 ** num_bits_per_symbol
            m = tf.math.real(m_mapper)  # self.mapper.constellation.points
            alpha = 1 / M * np.ones((num_tx_ant, M), dtype=GLOBAL_PRECISION)
        else:
            M = 2 ** int(num_bits_per_symbol / 2)
            c_poss = cmd_utils_tf2.tf_int2bin(np.array(
                range(0, 2 ** int(num_bits_per_symbol / 2))), int(num_bits_per_symbol / 2))
            b_power = 2 ** np.array(range(0, num_bits_per_symbol))[::-1][::2]
            ind_m = np.sum(b_power * c_poss, axis=-1)
            m = np.sqrt(2) * tf.cast(tf.math.real(tf.gather(m_mapper,
                                                            ind_m, axis=-1)), dtype=GLOBAL_PRECISION)
            alpha = 1 / M * np.ones((2 * num_tx_ant, M),
                                    dtype=GLOBAL_PRECISION)

        self.amp = AMP(num_iter, m, alpha, binary=self.binary,
                       gamma0=gamma0, delta0=delta0)
        self.symprob2llr = cmd_utils_tf2.TFSymprob2LLR(M, b=1)

    @tf.function  # (jit_compile = True)
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
            Ht = H_scale * tf.concat([tf.concat([tf.math.real(h), -tf.math.imag(
                h)], axis=-1), tf.concat([tf.math.imag(h), tf.math.real(h)], axis=-1)], axis=1)
            yt = y_scale * \
                tf.concat([tf.math.real(y), tf.math.imag(y)], axis=-1)
        # Real-valued
        ft, xt = self.amp([yt, Ht, sigmat0])
        llr_c, p_c = self.symprob2llr(ft)
        if not self.mod == 'pam':
            # llr_c = llr_c[..., tf.newaxis]
            # p_c = p_c[..., tf.newaxis]
            llr_c = tf.concat(
                [llr_c[:, :num_tx_ant], llr_c[:, num_tx_ant:]], axis=-1)
            llr_c = tf.concat([llr_c[..., 0::2], llr_c[..., 1::2]], axis=-1)
            p_c = tf.concat(
                [p_c[:, :num_tx_ant], p_c[:, num_tx_ant:]], axis=-1)
            p_c = tf.concat([p_c[..., 0::2], p_c[..., 1::2]], axis=-1)
            # ft = tf.reshape(tf.concat([ft[:, :model1_amp.num_tx_ant][..., tf.newaxis], ft[:, model1_amp.num_tx_ant:][..., tf.newaxis]], axis = -1), [-1, ft.shape[1], ft.shape[2]])
        return ft, xt, llr_c, p_c


class AMP(tf.keras.Model):
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

    def __init__(self, num_iter=64, m=np.array([-1, 1]), alpha=np.array([0.5, 0.5]), binary=False, gamma0=1, delta0=1,
                 **kwargs):
        super(AMP, self).__init__(**kwargs)
        self.it = tf.constant(num_iter)
        self.mt = m[tf.newaxis, tf.newaxis, ...]
        self.alpha = tf.constant(value=alpha)
        self.M = m.shape[0]
        gamma0 = tf.constant(gamma0, shape=(
            num_iter + 1), dtype=GLOBAL_PRECISION)
        delta0 = tf.constant(delta0, shape=(num_iter), dtype=GLOBAL_PRECISION)
        # Starting point
        self.s0 = tf.constant(tf.tensordot(alpha, m, axes=1)[
            tf.newaxis, :])  # a-priori mean
        self.amp_layer = []
        self.gamma = []
        self.delta = []
        for ii in range(num_iter + 1):
            if binary is True:
                if ii == 0:
                    self.amp_layer.append(AMPBinaryLayer(
                        m=self.mt, alpha=self.alpha, gamma=gamma0[ii], delta=delta0[ii], first_iter=True))
                elif ii == num_iter:
                    self.amp_layer.append(AMPBinaryLayer(
                        m=self.mt, alpha=self.alpha, gamma=gamma0[ii], delta=0, last_iter=True))
                else:
                    self.amp_layer.append(AMPBinaryLayer(
                        m=self.mt, alpha=self.alpha, gamma=gamma0[ii], delta=delta0[ii]))
            else:
                if ii == 0:
                    self.amp_layer.append(AMPLayer(
                        m=self.mt, alpha=self.alpha, gamma=gamma0[ii], delta=delta0[ii], first_iter=True))
                elif ii == num_iter:
                    self.amp_layer.append(AMPLayer(
                        m=self.mt, alpha=self.alpha, gamma=gamma0[ii], delta=0, last_iter=True))
                else:
                    self.amp_layer.append(
                        AMPLayer(m=self.mt, alpha=self.alpha, gamma=gamma0[ii], delta=delta0[ii]))
            self.gamma.append(self.amp_layer[-1].gamma)
            if not ii == num_iter:
                self.delta.append(self.amp_layer[-1].delta)

    @tf.function  # (jit_compile = True)
    def call(self, inputs):
        [yt, Ht, sigmat0] = inputs
        N0 = sigmat0 ** 2
        beta = Ht.shape[-1] / Ht.shape[-2]
        tau = beta * tf.reduce_mean(self.alpha * self.mt ** 2) / N0
        HH = tf.matmul(Ht, Ht, transpose_a=True)
        yH = tf.squeeze(tf.matmul(tf.expand_dims(yt, axis=1), Ht), axis=1)
        rH = yH - \
            tf.squeeze(tf.matmul(HH, tf.expand_dims(self.s0, axis=-1)), axis=-1)
        s = self.s0 * tf.ones_like(Ht[:, 0, :])
        ft = self.alpha
        for layer in self.amp_layer:
            s, rH, tau, ft = layer(s, rH, tau, HH, yH, N0, beta)
        # ft = w_m
        xt = s
        return ft, xt


class AMPLayer(tf.keras.layers.Layer):
    """The AMP layer
    """

    def __init__(self, m=np.array([-1, 1]), alpha=np.array([0.5, 0.5]), gamma=1, delta=1, first_iter=False, last_iter=False):
        super().__init__()
        # TODO: m and alpha trainable? -> needs to be changed in constellation object
        # and given to layers/function call
        self.mt = tf.constant(m)
        self.alpha = tf.constant(value=alpha)
        self.first_iter = first_iter
        self.last_iter = last_iter
        self.gamma = self.add_weight(shape=(1,),
                                     initializer=tf.keras.initializers.Constant(
            value=gamma),
            trainable=True,
            name='gamma')
        if not self.last_iter:
            self.delta = self.add_weight(shape=(1,),
                                         initializer=tf.keras.initializers.Constant(
                value=delta),
                trainable=True,
                name='delta')

    @tf.function  # (jit_compile = True)
    def call(self, s, rH, tau, HH, yH, N0, beta):
        # Start of new iteration
        tau = tau * self.gamma
        z = s + rH
        var_F = N0 * (1 + tau)
        arg = - 1 / 2 / var_F[:, tf.newaxis, tf.newaxis] * (
            z[..., tf.newaxis] - self.mt) ** 2 + tf.math.log(self.alpha[tf.newaxis, ...])
        w_m = tf.math.softmax(arg)
        s = tf.reduce_sum(w_m * self.mt, axis=-1)
        if not self.last_iter:
            G = tf.reduce_sum(
                w_m * (self.mt - s[..., tf.newaxis]) ** 2, axis=-1)
            tau_old = tau * self.delta
            tau = beta / N0 * tf.reduce_mean(G, axis=-1)
            rH = yH - self.delta * tf.squeeze(tf.matmul(HH, tf.expand_dims(
                s, axis=-1)), axis=-1) + (tau / (tau_old + 1))[:, tf.newaxis] * rH

        return s, rH, tau, w_m


class AMPBinaryLayer(tf.keras.layers.Layer):
    """The binary AMP layer
    """

    def __init__(self, m=np.array([-1, 1]), alpha=np.array([0.5, 0.5]), gamma=1, delta=1, first_iter=False, last_iter=False):
        super().__init__()
        # TODO: m and alpha trainable? -> needs to be changed in constellation object
        # and given to layers/function call
        self.mt = tf.constant(m)
        self.alpha = tf.constant(value=alpha)
        self.first_iter = first_iter
        self.last_iter = last_iter
        self.gamma = self.add_weight(shape=(1,),
                                     initializer=tf.keras.initializers.Constant(
            value=gamma),
            trainable=True,
            name='gamma')
        if not self.last_iter:
            self.delta = self.add_weight(shape=(1,),
                                         initializer=tf.keras.initializers.Constant(
                value=delta),
                trainable=True,
                name='delta')

    @tf.function  # (jit_compile = True)
    def call(self, s, rH, tau, HH, yH, N0, beta):
        # Start of new iteration
        tau = tau * self.gamma
        z = s + rH
        var_F = N0 * (1 + tau)
        s = tf.math.tanh(z / var_F[:, tf.newaxis])
        if not self.last_iter:
            tau_old = tau
            tau = beta / N0 * tf.reduce_mean(1 - s ** 2, axis=-1)
            rH = yH - self.delta * tf.squeeze(tf.matmul(HH, tf.expand_dims(
                s, axis=-1)), axis=-1) + (tau / (tau_old + 1))[:, tf.newaxis] * rH

        w_m = tf.concat([(1 + tf.math.sign(self.mt[..., 0]) * s[..., tf.newaxis]) / 2,
                         (1 + tf.math.sign(self.mt[..., 1]) * s[..., tf.newaxis]) / 2], axis=-1)
        return s, rH, tau, w_m
