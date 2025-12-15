#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 01 14:33:08 2022

CMDNet (Concrete MAP Detection Network) for MIMO - Keras layers

@author: beck
"""

# LOADED PACKAGES
# Tensorflow packages
import numpy as np
import tensorflow as tf
# Import Sionna
from sionna.mapping import Constellation
import cmdnet_utils_tf2 as cmd_utils_tf2

# Sionna only works with float32 precision in version 0.9.0
GLOBAL_PRECISION = 'float32'


class AlgoCMDNet(tf.keras.Model):
    """The CMDNet algorithm with llr output
    """

    def __init__(self, num_iter=64, const=Constellation('pam', 1), num_tx_ant=1, binary=False, taui0=1, delta0=1,
                 **kwargs):
        super(AlgoCMDNet, self).__init__(**kwargs)
        self.algo_name = 'CMDNet'
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
            m = np.sqrt(2) * tf.math.real(tf.gather(m_mapper, ind_m, axis=-1))
            alpha = 1 / M * np.ones((2 * num_tx_ant, M),
                                    dtype=GLOBAL_PRECISION)
            # Old index calculation for QAM modulation:
            # c_poss = tf_int2bin(np.array(range(0, 2 ** num_bits_per_symbol)), num_bits_per_symbol)
            # ind_m = mf.bin2int(c_poss[(c_poss[:, 1::2] == 0).all(axis = -1)], dtype = 'int64')

        if self.binary is True:
            self.bpsk = False       # TODO: Use True or False
            self.cmdnet = CMDNetBinary(
                num_iter, m, alpha, bpsk=self.bpsk, taui0=taui0, delta0=delta0)
        else:
            self.cmdnet = CMDNet(num_iter, m, alpha,
                                 taui0=taui0, delta0=delta0)
        self.symprob2llr = cmd_utils_tf2.TFSymprob2LLR(M, b=1)

    # Not using @tf.function at this point solves the NaN problem when training without performance loss. But why?
    # @tf.function#(jit_compile = True)
    def call(self, inputs):
        [y, h, sigmat0] = inputs
        num_tx_ant = tf.shape(h)[-1]
        num_rx_ant = tf.shape(h)[-2]
        y_scale = tf.cast(tf.sqrt(2 / num_rx_ant), dtype=GLOBAL_PRECISION)
        # Wrapper
        if self.mod == 'pam':
            Ht = y_scale * tf.cast(tf.math.real(h), dtype=GLOBAL_PRECISION)
            yt = y_scale * tf.cast(tf.math.real(y), dtype=GLOBAL_PRECISION)
        else:
            H_scale = tf.cast(tf.sqrt(2 / (2 * num_rx_ant)),
                              dtype=GLOBAL_PRECISION)
            Ht = H_scale * tf.cast(tf.concat([tf.concat([tf.math.real(h), -tf.math.imag(
                h)], axis=-1), tf.concat([tf.math.imag(h), tf.math.real(h)], axis=-1)], axis=1), dtype=GLOBAL_PRECISION)
            yt = y_scale * \
                tf.cast(tf.concat([tf.math.real(y), tf.math.imag(
                    y)], axis=-1), dtype=GLOBAL_PRECISION)
        # Real-valued
        ft, xt = self.cmdnet([yt, Ht, sigmat0])
        llr_c, p_c = self.symprob2llr(ft)
        if not self.mod == 'pam':
            llr_c = tf.concat(
                [llr_c[:, :num_tx_ant], llr_c[:, num_tx_ant:]], axis=-1)
            llr_c = tf.concat([llr_c[..., 0::2], llr_c[..., 1::2]], axis=-1)
            p_c = tf.concat(
                [p_c[:, :num_tx_ant], p_c[:, num_tx_ant:]], axis=-1)
            p_c = tf.concat([p_c[..., 0::2], p_c[..., 1::2]], axis=-1)
            # ft = tf.reshape(tf.concat([ft[:, :model1_cmdnet.num_tx_ant][..., tf.newaxis], ft[:, model1_cmdnet.num_tx_ant:][..., tf.newaxis]], axis = -1), [-1, ft.shape[1], ft.shape[2]])
        return ft, xt, llr_c, p_c


class CMDNet(tf.keras.Model):
    """The CMDNet algorithm
    """

    def __init__(self, num_iter=64, m=np.array([-1, 1]), alpha=np.array([0.5, 0.5]), taui0=1, delta0=1,
                 **kwargs):
        super(CMDNet, self).__init__(**kwargs)
        self.it = tf.constant(num_iter)
        # TODO: m and alpha trainable? -> needs to be changed in constellation object
        # and given to layers/function call
        self.mt = tf.expand_dims(tf.expand_dims(m, axis=0), axis=0)
        self.alpha = tf.constant(value=alpha)
        self.M = m.shape[0]
        taui0 = tf.constant(taui0, shape=(num_iter + 1),
                            dtype=GLOBAL_PRECISION)
        delta0 = tf.constant(delta0, shape=(num_iter), dtype=GLOBAL_PRECISION)
        self.G0 = tf.constant(value=np.zeros_like(alpha))
        self.cmd_layer = []
        self.taui = []
        self.delta = []
        for ii in range(num_iter + 1):
            if ii == 0:
                self.cmd_layer.append(CMDNetLayer(
                    m=self.mt, alpha=self.alpha, taui=taui0[ii], delta=delta0[ii], first_iter=True))
            elif ii == num_iter:
                self.cmd_layer.append(CMDNetLayer(
                    m=self.mt, alpha=self.alpha, taui=taui0[ii], delta=0, last_iter=True))
            else:
                self.cmd_layer.append(CMDNetLayer(
                    m=self.mt, alpha=self.alpha, taui=taui0[ii], delta=delta0[ii]))
            self.taui.append(self.cmd_layer[-1].taui)
            if not ii == num_iter:
                self.delta.append(self.cmd_layer[-1].delta)

    @tf.function  # (jit_compile = True)
    def call(self, inputs):
        [yt, Ht, sigmat0] = inputs
        HH = tf.matmul(Ht, Ht, transpose_a=True)
        yH = tf.squeeze(tf.matmul(tf.expand_dims(yt, axis=1), Ht), axis=1)
        G = tf.expand_dims(self.G0, axis=0) * \
            tf.expand_dims(tf.ones_like(Ht[:, 0, :]), axis=-1)
        sigmat = tf.expand_dims(tf.expand_dims(sigmat0, axis=-1), axis=-1)
        for layer in self.cmd_layer:
            G, ft, xt = layer(G, HH, yH, sigmat)
        return ft, xt


class CMDNetLayer(tf.keras.layers.Layer):
    """The CMDNet layer
    """

    def __init__(self, m=np.array([-1, 1]), alpha=np.array([0.5, 0.5]), taui=1, delta=1, first_iter=False, last_iter=False):
        super().__init__()
        # TODO: m and alpha trainable? -> needs to be changed in constellation object
        # and given to layers/function call
        self.mt = tf.constant(m)
        self.alpha = tf.constant(value=alpha)
        self.first_iter = first_iter
        self.last_iter = last_iter
        self.taui = self.add_weight(shape=(1,),
                                    initializer=tf.keras.initializers.Constant(
            value=taui),
            trainable=True,
            name='taui')
        if not self.last_iter:
            self.delta = self.add_weight(shape=(1,),
                                         initializer=tf.keras.initializers.Constant(
                value=delta),
                trainable=True,
                name='delta')

    @tf.function  # (jit_compile = True)
    def call(self, G, HH, yH, sigmat):
        # Start of new iteration
        taui_abs = tf.math.abs(self.taui)  # no negative values for tau !
        if self.first_iter is True:
            ft = tf.nn.softmax((tf.math.log(self.alpha) + G) * 1, axis=-1)
        else:
            ft = tf.nn.softmax((tf.math.log(self.alpha) + G)
                               * taui_abs, axis=-1)
        xt = tf.reduce_sum(ft * self.mt, axis=-1)

        if not self.last_iter:
            xHH = tf.squeeze(tf.matmul(tf.expand_dims(xt, axis=1), HH), axis=1)
            grad_x = taui_abs * (ft * self.mt - ft *
                                 tf.expand_dims(xt, axis=-1))
            # grad_L =  (1 - tf.math.exp(-G)) + 1 / sigmat ** 2 * grad_x * tf.expand_dims(xHH - yH, axis = -1) # original version
            grad_L = sigmat ** 2 * (1 - tf.math.exp(-G)) + \
                grad_x * tf.expand_dims(xHH - yH, axis=-1)
            # Gradient/ResNet Layer
            G = G - self.delta * grad_L
        return G, ft, xt


class CMDNetBinary(tf.keras.Model):
    """The binary CMDNet algorithm: bpsk or generic modulation
    """

    def __init__(self, num_iter=64, m=np.array([1, -1]), alpha=np.array([0.5, 0.5]), taui0=1, delta0=1, bpsk=True,
                 **kwargs):
        super(CMDNetBinary, self).__init__(**kwargs)
        self.it = num_iter
        # TODO: m and alpha trainable? -> needs to be changed in constellation object
        # and given to layers/function call
        self.mt = m  # tf.constant(m)
        # if (m == np.array([-1, 1])).all():
        #    alpha = alpha[:, 1]
        # else:
        alpha = alpha[:, 0]
        self.alpha = tf.constant(value=alpha)
        self.bpsk = bpsk
        self.M = m.shape[0]
        taui0 = tf.constant(taui0, shape=(num_iter + 1),
                            dtype=GLOBAL_PRECISION)
        delta0 = tf.constant(delta0, shape=(num_iter), dtype=GLOBAL_PRECISION)

        self.s0 = tf.constant(value=np.zeros_like(alpha))
        self.cmd_layer = []
        self.taui = []
        self.delta = []
        for ii in range(num_iter + 1):
            if self.bpsk is True:
                if ii == 0:
                    self.cmd_layer.append(CMDNetBinaryLayer(
                        m=self.mt, alpha=self.alpha, taui=taui0[ii], delta=delta0[ii], first_iter=True))
                elif ii == num_iter:
                    self.cmd_layer.append(CMDNetBinaryLayer(
                        m=self.mt, alpha=self.alpha, taui=taui0[ii], delta=0, last_iter=True))
                else:
                    self.cmd_layer.append(CMDNetBinaryLayer(
                        m=self.mt, alpha=self.alpha, taui=taui0[ii], delta=delta0[ii]))
            else:
                if ii == 0:
                    self.cmd_layer.append(CMDNetGenericBinaryLayer(
                        m=self.mt, alpha=self.alpha, taui=taui0[ii], delta=delta0[ii], first_iter=True))
                elif ii == num_iter:
                    self.cmd_layer.append(CMDNetGenericBinaryLayer(
                        m=self.mt, alpha=self.alpha, taui=taui0[ii], delta=0, last_iter=True))
                else:
                    self.cmd_layer.append(CMDNetGenericBinaryLayer(
                        m=self.mt, alpha=self.alpha, taui=taui0[ii], delta=delta0[ii]))
            self.taui.append(self.cmd_layer[-1].taui)
            if not ii == num_iter:
                self.delta.append(self.cmd_layer[-1].delta)

    @tf.function  # (jit_compile = True)
    def call(self, inputs):
        [yt, Ht, sigmat0] = inputs
        HH = tf.matmul(Ht, Ht, transpose_a=True)
        yH = tf.squeeze(tf.matmul(tf.expand_dims(yt, axis=1), Ht), axis=1)
        s = tf.transpose(tf.expand_dims(self.s0, axis=-1)) * \
            tf.ones_like(Ht[:, 0, :])
        sigmat = tf.expand_dims(sigmat0, axis=-1)
        for layer in self.cmd_layer:
            s, ft, xt = layer(s, HH, yH, sigmat)
        return ft, xt


class CMDNetBinaryLayer(tf.keras.layers.Layer):
    """The binary CMDNet layer
    """

    def __init__(self, m=np.array([1, -1]), alpha=np.array([0.5, 0.5]), taui=1, delta=1, first_iter=False, last_iter=False):
        super().__init__()
        # TODO: m and alpha trainable? -> needs to be changed in constellation object
        # and given to layers/function call
        self.m = m  # tf.constant(m)
        self.alpha = tf.constant(alpha)
        self.first_iter = first_iter
        self.last_iter = last_iter
        self.taui = self.add_weight(shape=(1,),
                                    initializer=tf.keras.initializers.Constant(
            value=taui),
            trainable=True,
            name='taui')
        if not self.last_iter:
            self.delta = self.add_weight(shape=(1,),
                                         initializer=tf.keras.initializers.Constant(
                value=delta),
                trainable=True,
                name='delta')

    @tf.function  # (jit_compile = True)
    def call(self, s, HH, yH, sigmat):
        # Start of new iteration
        taui_abs = tf.math.abs(self.taui)  # no negative values for tau !
        if self.first_iter is True:
            xt = tf.math.tanh((tf.math.log(1 / self.alpha - 1) + s) / 2 * 1)
        else:
            xt = tf.math.tanh(
                (tf.math.log(1 / self.alpha - 1) + s) / 2 * taui_abs)
        xt2 = tf.expand_dims(xt, axis=-1)
        # if tf.math.reduce_all(self.m == np.array([-1, 1])):
        #     ft = tf.concat([(1 - xt2) / 2, (1 + xt2) / 2], axis = -1) # [q(x = -1), q(x = 1)]
        # else:
        #     ft = tf.concat([(1 + xt2) / 2, (1 - xt2) / 2], axis = -1) # [q(x = 1), q(x = -1)]
        # [q(x = m_1), q(x = m_2)]
        ft = tf.concat([(1 + self.m[0] * xt2) / 2,
                        (1 + self.m[1] * xt2) / 2], axis=-1)

        if not self.last_iter:
            xHH = tf.squeeze(tf.matmul(tf.expand_dims(xt, axis=1), HH), axis=1)
            grad_x = 1 / 2 * taui_abs * (1 - xt ** 2)
            grad_L = sigmat ** 2 * tf.math.tanh(s / 2) + grad_x * (xHH - yH)
            # grad_L = tf.math.tanh(s / 2) + 1 / sigmat ** 2 * grad_x * (xHH - yH) # original version
            # Gradient/ResNet Layer
            s = s - self.delta * grad_L
        return s, ft, xt


class CMDNetGenericBinaryLayer(tf.keras.layers.Layer):
    """The generic binary CMDNet layer: New general case where we have 2 arbitrary symbols in m
    """

    def __init__(self, m=np.array([1, -1]), alpha=np.array([0.5, 0.5]), taui=1, delta=1, first_iter=False, last_iter=False):
        super().__init__()
        # TODO: m and alpha trainable? -> needs to be changed in constellation object
        # and given to layers/function call
        self.m = tf.constant(tf.cast(m, dtype=GLOBAL_PRECISION))
        self.alpha = tf.constant(alpha)
        self.first_iter = first_iter
        self.last_iter = last_iter
        self.taui = self.add_weight(shape=(1,),
                                    initializer=tf.keras.initializers.Constant(
            value=taui),
            trainable=True,
            name='taui')
        if not self.last_iter:
            self.delta = self.add_weight(shape=(1,),
                                         initializer=tf.keras.initializers.Constant(
                value=delta),
                trainable=True,
                name='delta')

    @tf.function  # (jit_compile = True)
    def call(self, s, HH, yH, sigmat):
        # Start of new iteration
        taui_abs = tf.math.abs(self.taui)  # no negative values for tau !
        if self.first_iter is True:
            ft0 = tf.math.sigmoid(-(tf.math.log(1 / self.alpha - 1) + s) * 1)
        else:
            ft0 = tf.math.sigmoid(-(tf.math.log(1 /
                                                self.alpha - 1) + s) * taui_abs)
        ft = tf.concat([ft0[..., tf.newaxis], (1 - ft0)
                        [..., tf.newaxis]], axis=-1)
        xt = tf.cast(ft0, dtype=GLOBAL_PRECISION) * \
            (self.m[0] - self.m[1]) + self.m[1]

        if not self.last_iter:
            xHH = tf.squeeze(tf.matmul(tf.expand_dims(xt, axis=1), HH), axis=1)
            grad_x = -taui_abs * ft0 * (1 - ft0) * (self.m[0] - self.m[1])
            grad_L = sigmat ** 2 * tf.math.tanh(s / 2) + grad_x * (xHH - yH)
            # grad_L = tf.math.tanh(s / 2) + 1 / sigmat ** 2 * grad_x * (xHH - yH) # original version
            # Gradient/ResNet Layer
            s = s - self.delta * grad_L
        return s, ft, xt
