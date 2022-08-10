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
from sionna.mapping import SymbolDemapper, Mapper, Demapper
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import myfunctions as mf
from tensorflow.keras import backend as KB
import tikzplotlib as tplt
from tensorflow.keras.models import Model


# sn.config.xla_compat = True
class Model(tf.keras.Model):
    def __init__(self, spatial_corr = None, algo = 1):
        super().__init__()
        self.n = 1024 
        self.k = 512  
        self.coderate = self.k / self.n
        self.num_bits_per_symbol = 1
        self.num_tx_ant = 64
        self.num_rx_ant = 64
        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.mapper = Mapper("pam", self.num_bits_per_symbol)
        self.demapper = Demapper("app", "pam", self.num_bits_per_symbol)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out = True)
        self.channel = FlatFadingChannel(self.num_tx_ant,
                                         self.num_rx_ant,
                                         spatial_corr = spatial_corr,
                                         add_awgn = True,
                                         return_channel = True)
        # Load starting point
        sim_set = {
            'Mod': 'QPSK',
            'Nr': 64,
            'Nt': 64,
            'L': 64,
        }
        saveobj2 = mf.savemodule('npz')
        fn2 = mf.filename_module('trainhist_', 'curves', 'CMD', '_binary_tau0.1', sim_set) # _binary_tau0.1, _tau0.1
        train_hist2 = mf.training_history()
        train_hist2.dict2obj(saveobj2.load(fn2.pathfile))
        [delta0, taui0] = train_hist2.params[-1]
        M = 2
        alpha = 1 / M * np.ones((self.num_tx_ant, M), dtype = 'float32')
        m = np.array([1, -1], dtype = 'float32')
        self.cmdnet0 = cmdnet_bin2(64, m, alpha, taui0 = taui0, delta0 = delta0)
        # self.cmdnet0 = cmdnet(64, m, alpha, taui0 = taui0, delta0 = delta0)
        self.algo = algo # choose algorithm
        
    # @tf.function #(jit_compile=True)
    def call(self, batch_size, ebno_db, ebno_db_max = None):
        b = self.binary_source([batch_size, self.num_tx_ant, self.k])
        c = b #self.encoder(b)
        shapec = tf.shape(c)

        x = self.mapper(c)
        shape = tf.shape(x)
        x = tf.reshape(x, [-1, self.num_tx_ant])
        
        if ebno_db_max == None:
            ebno_db_max = ebno_db
        ebno_db_vec = tf.random.uniform(shape = [x.shape[0]], minval = ebno_db, maxval = ebno_db_max)
        no = 1 / 2 * ebnodb2no(ebno_db_vec, self.num_bits_per_symbol, 1) # self.coderate)
        
        if self.algo == 1:
            no_scale = self.num_rx_ant * 2 / 2  # scaling of noise variance necessary since channel has Rayleigh taps with variance 1
            # no *= np.sqrt(self.num_rx_ant)    # original, why square root?

            y, h = self.channel([x, no * no_scale])
            s = tf.complex(no * tf.eye(self.num_rx_ant, self.num_rx_ant), 0.0)
            H_scale = np.sqrt(2 / self.num_rx_ant)
            # x_hat, no_eff = lmmse_equalizer(y * H_scale, h * H_scale, s)
            x_hat, no_eff = lmmse_equalizer(tf.cast(tf.math.real(y) * H_scale, dtype = 'complex64'), tf.cast(tf.math.real(h) * H_scale, dtype = 'complex64'), s)
            # sigmat0 = tf.math.sqrt(no) # * no_scale)
            # x_hat, Phi_ee = mf.mmse([tf.math.real(y) * H_scale, tf.math.real(h) * H_scale, tf.repeat(sigmat0, y.shape[0])])
            # no_eff = mf.tdiag2vec(Phi_ee)
        
            x_hat = tf.reshape(x_hat, shape)
            no_eff = tf.reshape(no_eff, shape)
        
            # # TODO: Demapper is not valid for probs!!!
            llr = self.demapper([tf.cast(x_hat, dtype = 'complex64'), tf.cast(no_eff, dtype = 'float32')])
            # llr = self.demapper([x_hat, no_eff])
            # b_hat = self.decoder(llr)
        else:
            no_scale = self.num_rx_ant * 2 / 2  # scaling of noise variance necessary since channel has Rayleigh taps with variance 1
            # no *= np.sqrt(self.num_rx_ant)    # original, why square root?

            y, h = self.channel([x, no * no_scale])
            # s = tf.complex(no * tf.eye(self.num_rx_ant, self.num_rx_ant), 0.0)
            # Alternative: CMDNet
            # Complex ?
            # Ht = tf.concat([tf.concat([tf.math.real(h), tf.math.imag(h)], axis = -1), tf.concat([-tf.math.imag(h), tf.math.real(h)], axis = -1)], axis = -2) # / np.sqrt(model1.num_rx_ant)
            # yt = tf.concat([tf.math.real(y), tf.math.imag(y)], axis = -1)
            # M = 2 ** model1.num_bits_per_symbol
            # Ht = h # tf.transpose(h, [0, 2, 1])
            # HHy = tf.squeeze(tf.matmul(tf.linalg.adjoint(Ht), tf.expand_dims(y, axis = -1)), axis = -1)
            # HHH = tf.matmul(tf.linalg.adjoint(Ht), Ht)
            # Equalization
            # M = 2 # int(np.sqrt(2 ** model1.num_bits_per_symbol)) # we split real and imaginary part
            # alpha = 1 / M * np.ones((self.num_tx_ant, M), dtype = 'float32')
            # m = np.concatenate([tf.math.real(model1.mapper.constellation.points)[0::8].numpy(), tf.math.real(model1.mapper.constellation.points)[3::8].numpy()], axis = -1, dtype = 'float32')
            # m = np.array([1, -1], dtype = 'float32')
            # cmdnet0 = cmdnet_bin(alpha, m, 16)
            # Real-valued
            H_scale = np.sqrt(2 / self.num_rx_ant)
            Ht = tf.math.real(h) * H_scale
            yt = tf.math.real(y) * H_scale
            sigmat0 = tf.math.sqrt(no)
            ft, xt = self.cmdnet0([yt, Ht, sigmat0])
            xtc = xt
            # # LLR Demapping
            # # wenn das klappt: alles zusammenfügen
            # # TODO: object daraus machen mit konstanter maske
            # llr_c, _ = tf_symprob2llr(ft, M)
            # llr = tf.reshape(llr_c, shapec)
            # xtc = tf.cast(xt[:, :int(xt.shape[-1] / 2)], dtype = 'complex64') + 1j * tf.cast(xt[:, int(xt.shape[-1] / 2):], dtype = 'complex64')
            # xtc = tf.reshape(xtc, shape)
            llr = self.demapper([tf.cast(xtc, dtype = 'complex64'), tf.ones(xtc.shape, dtype = 'float32')])
        b_hat = tf.cast(tf.reshape(llr > 0, shapec), dtype = 'float32')

        # output layer and objective function
        soft = 0
        if soft == 1:
            # 2. mean should be a sum for overall MSE scaling with Nt
            loss = KB.mean(KB.mean((x - xt) ** 2, axis = -1))
        else:
            # 2. mean should be a sum since q factorizes
            cl = tf.reshape(b, [-1, self.num_tx_ant, 1])
            cl = tf.concat([cl, 1 - cl], axis = -1)
            loss = KB.mean(KB.mean(KB.categorical_crossentropy(cl, ft, axis = -1), axis = -1))

        return b, b_hat, loss



def tf_symprob2llr(p_m, M):
    '''Calculate llrs of bits c from symbol probabilities p_m of modulation alphabet of cardinality M
    p_m: symbol probabilities p_m
    M: Modulation order/ number of symbols
    llr_c: llr of code bit c; ln(p(c=0)/p(c=1)) = ln(p(c=0)/(1-p(c=0)))
    p_c0: probability of code bit c being 0; p(c=0)
    '''
    # constant
    c_poss = mf.int2bin(np.array(range(0, M)), int(np.log2(M)))
    mask = (c_poss == 0)[np.newaxis, np.newaxis, ...]
    # tensorflow
    p_c0 = tf.reduce_sum(p_m[..., tf.newaxis] * mask, axis = -2)
    llr_c = tf.math.log(p_c0 / (1 - p_c0))
    llr_c = tf.clip_by_value(llr_c, -1e6, 1e6)   # avoid infinity
    return llr_c, p_c0


class cmdnet3(tf.keras.layers.Layer):
    '''CMDNet layer
    '''
    def __init__(self, num_iter = 64, m = np.array([-1, 1]), alpha = np.array([0.5, 0.5]), taui0 = 1, delta0 = 1):
        super(cmdnet3, self).__init__()
        self.it = tf.constant(num_iter)
        self.alphat = tf.constant(value = alpha)
        self.m = tf.constant(value = m)
        self.M = m.shape[0]
        #func_tau = tau0 * np.ones(self.it + 1)
        self.taui = self.add_weight(shape = (num_iter + 1,),
                             initializer = tf.keras.initializers.Constant(value = taui0),
                             trainable = True)
        self.delta = self.add_weight(shape = (num_iter,),
                             initializer = tf.keras.initializers.Constant(value = delta0),
                             trainable = True)
        # self.G0 = self.add_weight(shape = alpha.shape,
        #                      initializer=tf.keras.initializers.Constant(value = np.zeros_like(alpha)),
        #                      trainable=False)
        # self.taui = tf.Variable(initial_value = 1 / func_tau,
        #                         trainable = True)
        # self.delta = tf.Variable(initial_value = delta0 * np.ones(it),
        #                          trainable = True)
        self.G0 = tf.constant(value = np.zeros_like(alpha))
        #b_init = tf.zeros_initializer()
        #self.b = tf.Variable(initial_value=b_init(shape=(units,),
        #                                          dtype='float32'),
        #                     trainable=True)
    
    # @tf.function
    def call(self, inputs):
        # inputs = [G, y, Hr, sigma]
        [yt, Ht, sigmat0] = inputs
        sigmat = tf.expand_dims(tf.expand_dims(sigmat0, axis = -1), axis = -1)
        
        G = KB.expand_dims(self.G0, axis = 0) * KB.expand_dims(KB.ones_like(Ht[:, 0, :]), axis = -1)
        # G = tf.expand_dims(self.G0, axis = 0) * tf.expand_dims(tf.ones_like(Ht[:, 0, :]), axis = -1)
        taui_abs = KB.abs(self.taui[0])
        # taui_abs = tf.math.abs(self.taui[0]) # no negative values for tau !
        # ft = tf.math.softmax((tf.math.log(self.alphat) + G) * tf.math.abs(taui_abs), axis = -1)
        ft = KB.softmax((KB.log(self.alphat) + G) * taui_abs, axis = -1)
        # mt = tf.expand_dims(tf.expand_dims(self.m, axis = 0), axis = 0)
        mt = KB.expand_dims(KB.expand_dims(self.m, axis = 0), axis = 0)
        xt = KB.sum(ft * mt, axis = -1)
        # xt = tf.reduce_sum(ft * self.m, axis = -1)
        
        # UNFOLDING
        HH = KB.batch_dot(KB.permute_dimensions(Ht, (0, 2, 1)), Ht)
        yH = KB.batch_dot(yt, Ht)
        # HTy = tf.squeeze(tf.matmul(tf.transpose(Ht, [0, 2, 1]), tf.expand_dims(yt, axis = -1)), axis = -1)
        # HTH = tf.matmul(tf.transpose(Ht, [0, 2, 1]), Ht)
        for iteration in tf.range(0, self.it):
            # HTHx = tf.squeeze(tf.matmul(HTH, tf.expand_dims(xt, axis = -1)), axis = -1)
            xHH = KB.batch_dot(xt, HH)
            # grad_x = taui_abs * (ft * mt - ft * tf.expand_dims(xt, axis = -1))
            grad_x = taui_abs * (ft * mt- ft * KB.expand_dims(xt, axis = -1))
            # grad_L = sigmat ** 2 * (1 - tf.math.exp(-G)) + grad_x * tf.expand_dims(HTHx - HTy, axis = -1)
            # grad_L =  (1 - KB.exp(-G)) + 1 / sigmat ** 2 * grad_x * KB.expand_dims(xHH - yH, axis = -1) # original version
            grad_L = sigmat ** 2 * (1 - KB.exp(-G)) + grad_x * KB.expand_dims(xHH - yH, axis = -1)
            # Gradient/ResNet Layer
            G = G - self.delta[iteration] * grad_L
            # Start of new iteration
            taui_abs = KB.abs(self.taui[iteration + 1]) # no negative values for tau !
            #taui_abs = tf.math.abs(self.taui[iteration + 1]) # no negative values for tau !
            #logits = (KB.log(self.alphat) + G) * taui_abs
            #logits = (tf.math.log(self.alphat) + G) * taui_abs
            # Softmax with complex values???
            # Real- und Imaginärteil haben eigene Wahrscheinlichkeiten??? Summe über Real- und Imaginärteil separat
            # ft = tf.math.exp(logits) / tf.reduce_sum(tf.math.exp(logits), axis = -1)[..., tf.newaxis]
            ft = KB.softmax((KB.log(self.alphat) + G) * taui_abs, axis = -1)
            #ft = tf.math.softmax(logits, axis = -1)
            xt = KB.sum(ft * mt, axis = -1)
            #xt = tf.reduce_sum(ft * self.m, axis = -1)
        return ft, xt


class cmdnet(tf.keras.layers.Layer):
    '''CMDNet layer
    '''
    def __init__(self, m, alpha, it, taui0 = 1, delta0 = 1):
        super(cmdnet, self).__init__()
        self.it = tf.constant(it)
        self.alphat = tf.constant(value = alpha)
        self.m = tf.constant(value = m)
        self.M = m.shape[0]
        #func_tau = tau0 * np.ones(self.it + 1)
        self.taui = self.add_weight(shape = (it + 1,),
                             initializer = tf.keras.initializers.Constant(value = taui0),
                             trainable = True)
        self.delta = self.add_weight(shape = (it,),
                             initializer = tf.keras.initializers.Constant(value = delta0),
                             trainable = True)
        # self.G0 = self.add_weight(shape = alpha.shape,
        #                      initializer=tf.keras.initializers.Constant(value = np.zeros_like(alpha)),
        #                      trainable=False)
        # self.taui = tf.Variable(initial_value = 1 / func_tau,
        #                         trainable = True)
        # self.delta = tf.Variable(initial_value = delta0 * np.ones(it),
        #                          trainable = True)
        self.G0 = tf.constant(value = np.zeros_like(alpha))
        #b_init = tf.zeros_initializer()
        #self.b = tf.Variable(initial_value=b_init(shape=(units,),
        #                                          dtype='float32'),
        #                     trainable=True)
    
    # @tf.function
    def call(self, inputs):
        # inputs = [G, y, Hr, sigma]
        [yt, Ht, sigmat0] = inputs
        sigmat = tf.expand_dims(tf.expand_dims(sigmat0, axis = -1), axis = -1)
        
        G = KB.expand_dims(self.G0, axis = 0) * KB.expand_dims(KB.ones_like(Ht[:, 0, :]), axis = -1)
        # G = tf.expand_dims(self.G0, axis = 0) * tf.expand_dims(tf.ones_like(Ht[:, 0, :]), axis = -1)
        taui_abs = KB.abs(self.taui[0])
        # taui_abs = tf.math.abs(self.taui[0]) # no negative values for tau !
        # ft = tf.math.softmax((tf.math.log(self.alphat) + G) * tf.math.abs(taui_abs), axis = -1)
        ft = KB.softmax((KB.log(self.alphat) + G) * taui_abs, axis = -1)
        # mt = tf.expand_dims(tf.expand_dims(self.m, axis = 0), axis = 0)
        mt = KB.expand_dims(KB.expand_dims(self.m, axis = 0), axis = 0)
        xt = KB.sum(ft * mt, axis = -1)
        # xt = tf.reduce_sum(ft * self.m, axis = -1)
        
        # UNFOLDING
        HH = KB.batch_dot(KB.permute_dimensions(Ht, (0, 2, 1)), Ht)
        yH = KB.batch_dot(yt, Ht)
        # HTy = tf.squeeze(tf.matmul(tf.transpose(Ht, [0, 2, 1]), tf.expand_dims(yt, axis = -1)), axis = -1)
        # HTH = tf.matmul(tf.transpose(Ht, [0, 2, 1]), Ht)
        for iteration in tf.range(0, self.it):
            # HTHx = tf.squeeze(tf.matmul(HTH, tf.expand_dims(xt, axis = -1)), axis = -1)
            xHH = KB.batch_dot(xt, HH)
            # grad_x = taui_abs * (ft * mt - ft * tf.expand_dims(xt, axis = -1))
            grad_x = taui_abs * (ft * mt- ft * KB.expand_dims(xt, axis = -1))
            # grad_L = sigmat ** 2 * (1 - tf.math.exp(-G)) + grad_x * tf.expand_dims(HTHx - HTy, axis = -1)
            # grad_L =  (1 - KB.exp(-G)) + 1 / sigmat ** 2 * grad_x * KB.expand_dims(xHH - yH, axis = -1) # original version
            grad_L = sigmat ** 2 * (1 - KB.exp(-G)) + grad_x * KB.expand_dims(xHH - yH, axis = -1)
            # Gradient/ResNet Layer
            G = G - self.delta[iteration] * grad_L
            # Start of new iteration
            taui_abs = KB.abs(self.taui[iteration + 1]) # no negative values for tau !
            #taui_abs = tf.math.abs(self.taui[iteration + 1]) # no negative values for tau !
            #logits = (KB.log(self.alphat) + G) * taui_abs
            #logits = (tf.math.log(self.alphat) + G) * taui_abs
            # Softmax with complex values???
            # Real- und Imaginärteil haben eigene Wahrscheinlichkeiten??? Summe über Real- und Imaginärteil separat
            # ft = tf.math.exp(logits) / tf.reduce_sum(tf.math.exp(logits), axis = -1)[..., tf.newaxis]
            ft = KB.softmax((KB.log(self.alphat) + G) * taui_abs, axis = -1)
            #ft = tf.math.softmax(logits, axis = -1)
            xt = KB.sum(ft * mt, axis = -1)
            #xt = tf.reduce_sum(ft * self.m, axis = -1)
        return ft, xt


class cmdnet_bin_layer(tf.keras.layers.Layer):
    """The binary CMDNet layer
    """
    def __init__(self, m = np.array([-1, 1]), alpha = np.array([0.5, 0.5]), taui = 1, delta = 1, first_iter = False):
        super().__init__()
        self.m = m
        self.alpha = alpha
        self.first_iter = first_iter
        # self.taui = taui
        # self.delta = delta
        self.taui = self.add_weight(shape = (1,),
                             initializer = tf.keras.initializers.Constant(value = taui),
                             trainable = True,
                             name = 'taui')
        self.delta = self.add_weight(shape = (1,),
                             initializer = tf.keras.initializers.Constant(value = delta),
                             trainable = True,
                             name = 'delta')

    def call(self, s, HH, yH, sigmat):
        # Start of new iteration
        taui_abs = KB.abs(self.taui) # no negative values for tau !
        if self.first_iter == True:
            xt = KB.tanh((KB.log(1 / self.alpha - 1) + s) / 2 * 1)
        else:
            xt = KB.tanh((KB.log(1 / self.alpha - 1) + s) / 2 * taui_abs)
        xt2 = KB.expand_dims(xt, axis = -1)
        if (self.m == np.array([-1, 1])).all():
            ft = KB.concatenate([(1 - xt2) / 2, (1 + xt2) / 2], axis = -1) # [q(x = -1), q(x = 1)]
        else:
            ft = KB.concatenate([(1 + xt2) / 2, (1 - xt2) / 2], axis = -1) # [q(x = 1), q(x = -1)]
        
        xHH = KB.batch_dot(xt, HH)
        grad_x = 1 / 2 * taui_abs * (1 - xt ** 2) 
        grad_L = sigmat ** 2 * KB.tanh(s / 2) + grad_x * (xHH - yH)
        # grad_L = KB.tanh(s / 2) + 1 / sigmat ** 2 * grad_x * (xHH - yH) # original version
        # Gradient/ResNet Layer
        s = s - self.delta * grad_L
        return s


class cmdnet_bin2(tf.keras.Model):
    """The binary CMDNet algorithm
    """
    def __init__(self, num_iter = 64, m = np.array([-1, 1]), alpha = np.array([0.5, 0.5]), taui0 = 1, delta0 = 1,
                    **kwargs):
        super(cmdnet_bin2, self).__init__(**kwargs)
        self.it = tf.constant(num_iter)
        self.m = m
        self.M = m.shape[0]
        if (m == np.array([-1, 1])).all():
            alpha = alpha[:, 0]
        else:
            alpha = alpha[:, 1]
        self.alpha = tf.constant(value = alpha)
        # self.taui = self.add_weight(shape = (num_iter + 1,),
        #                      initializer = tf.keras.initializers.Constant(value = taui0),
        #                      trainable = True,
        #                      name = 'taui')
        # self.delta = self.add_weight(shape = (num_iter,),
        #                      initializer = tf.keras.initializers.Constant(value = delta0),
        #                      trainable = True,
        #                      name = 'delta')
        taui0 = tf.constant(taui0, shape = (num_iter + 1), dtype = 'float32')
        delta0 = tf.constant(delta0, shape = (num_iter), dtype = 'float32')
        
        self.s0 = tf.constant(value = np.zeros_like(alpha))
        self.cmdnet = []
        self.taui = []
        self.delta = []
        for ii in range(num_iter):
            if ii == 0:
                self.cmdnet.append(cmdnet_bin_layer(m = self.m, alpha = self.alpha, taui = taui0[ii], delta = delta0[ii], first_iter = True))
            else:
                self.cmdnet.append(cmdnet_bin_layer(m = self.m, alpha = self.alpha, taui = taui0[ii], delta = delta0[ii], first_iter = False))
            self.taui.append(self.cmdnet[-1].taui)
            self.delta.append(self.cmdnet[-1].delta)


    def call(self, inputs):
        [yt, Ht, sigmat0] = inputs
        HH = KB.batch_dot(KB.permute_dimensions(Ht, (0, 2, 1)), Ht)
        yH = KB.batch_dot(yt, Ht)
        s = KB.transpose(KB.expand_dims(self.s0)) * KB.ones_like(Ht[:, 0, :])
        sigmat = KB.expand_dims(sigmat0, axis = -1)
        for layer in self.cmdnet:
            s = layer(s, HH, yH, sigmat)
        # Start of last iteration
        taui_abs = KB.abs(self.taui[-1]) # no negative values for tau !
        xt = KB.tanh((KB.log(1 / self.alpha - 1) + s) / 2 * taui_abs)
        xt2 = KB.expand_dims(xt, axis = -1)
        if (self.m == np.array([-1, 1])).all():
            ft = KB.concatenate([(1 - xt2) / 2, (1 + xt2) / 2], axis = -1) # [q(x = -1), q(x = 1)]
        else:
            ft = KB.concatenate([(1 + xt2) / 2, (1 - xt2) / 2], axis = -1) # [q(x = 1), q(x = -1)]
        return ft, xt



class cmdnet_bin(tf.keras.layers.Layer):
    '''Binary CMDNet layer
    '''
    def __init__(self, it, m, alpha, taui0 = 1, delta0 = 1):
        super(cmdnet_bin, self).__init__()
        self.it = tf.constant(it)
        self.m = m
        self.M = m.shape[0]
        if (m == np.array([-1, 1])).all():
            alpha = alpha[:, 0]
        else:
            alpha = alpha[:, 1]
        self.alphat = tf.constant(value = alpha)
        #func_tau = tau0 * np.ones(self.it + 1)
        self.taui = self.add_weight(shape = (it + 1,),
                             initializer = tf.keras.initializers.Constant(value = taui0),
                             trainable = True)
        self.delta = self.add_weight(shape = (it,),
                             initializer = tf.keras.initializers.Constant(value = delta0),
                             trainable = True)
        self.s0 = tf.constant(value = np.zeros_like(alpha))
        # self.G0 = self.add_weight(shape = alpha.shape,
        #                      initializer=tf.keras.initializers.Constant(value = np.zeros_like(alpha)),
        #                      trainable=False)
        # self.taui = tf.Variable(initial_value = 1 / func_tau,
        #                         trainable = True)
        # self.delta = tf.Variable(initial_value = delta0 * np.ones(it),
        #                          trainable = True)
        #b_init = tf.zeros_initializer()
        #self.b = tf.Variable(initial_value=b_init(shape=(units,),
        #                                          dtype='float32'),
        #                     trainable=True)
    
    # @tf.function
    def call(self, inputs):
        [yt, Ht, sigmat0] = inputs
        sigmat = tf.expand_dims(sigmat0, axis = -1)
        
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
            taui_abs = KB.abs(self.taui[iteration + 1]) # no negative values for tau !
            xt = KB.tanh((KB.log(1 / alphat - 1) + s) / 2 * taui_abs)
            xt2 = KB.expand_dims(xt, axis = -1)
            if (self.m == np.array([-1, 1])).all():
                ft = KB.concatenate([(1 - xt2) / 2, (1 + xt2) / 2], axis = -1) # [q(x = -1), q(x = 1)]
            else:
                ft = KB.concatenate([(1 + xt2) / 2, (1 - xt2) / 2], axis = -1) # [q(x = 1), q(x = -1)]
        return ft, xt


# Neue Funktionen

class cmdnet_kb(tf.keras.Model):
    """The CMDNet algorithm
    """
    def __init__(self, num_iter = 64, m = np.array([-1, 1]), alpha = np.array([0.5, 0.5]), taui0 = 1, delta0 = 1,
                    **kwargs):
        super(cmdnet, self).__init__(**kwargs)
        self.it = tf.constant(num_iter)
        self.mt = KB.expand_dims(KB.expand_dims(m, axis = 0), axis = 0)
        self.M = m.shape[0]
        self.alpha = tf.constant(value = alpha)
        taui0 = tf.constant(taui0, shape = (num_iter + 1), dtype = 'float32')
        delta0 = tf.constant(delta0, shape = (num_iter), dtype = 'float32')
        self.G0 = tf.constant(value = np.zeros_like(alpha))
        self.cmd_layer = []
        self.taui = []
        self.delta = []
        for ii in range(num_iter):
            if ii == 0:
                self.cmd_layer.append(cmdnet_layer_kb(m = self.mt, alpha = self.alpha, taui = taui0[ii], delta = delta0[ii], first_iter = True))
            else:
                self.cmd_layer.append(cmdnet_layer_kb(m = self.mt, alpha = self.alpha, taui = taui0[ii], delta = delta0[ii], first_iter = False))
            self.taui.append(self.cmd_layer[-1].taui)
            self.delta.append(self.cmd_layer[-1].delta)


    def call(self, inputs):
        [yt, Ht, sigmat0] = inputs
        HH = KB.batch_dot(KB.permute_dimensions(Ht, (0, 2, 1)), Ht)
        yH = KB.batch_dot(yt, Ht)
        G = KB.expand_dims(self.G0, axis = 0) * KB.expand_dims(KB.ones_like(Ht[:, 0, :]), axis = -1)
        sigmat = KB.expand_dims(KB.expand_dims(sigmat0, axis = -1), axis = -1)
        for layer in self.cmd_layer:
            G = layer(G, HH, yH, sigmat)
        # Start of last iteration
        taui_abs = KB.abs(self.taui[-1]) # no negative values for tau !
        ft = KB.softmax((KB.log(self.alpha) + G) * taui_abs, axis = -1)
        xt = KB.sum(ft * self.mt, axis = -1)
        return ft, xt

class cmdnet_layer_kb(tf.keras.layers.Layer):
    """The CMDNet layer
    """
    def __init__(self, m = np.array([-1, 1]), alpha = np.array([0.5, 0.5]), taui = 1, delta = 1, first_iter = False):
        super().__init__()
        self.mt = m
        self.alpha = tf.constant(value = alpha)
        self.first_iter = first_iter
        self.taui = self.add_weight(shape = (1,),
                             initializer = tf.keras.initializers.Constant(value = taui),
                             trainable = True,
                             name = 'taui')
        self.delta = self.add_weight(shape = (1,),
                             initializer = tf.keras.initializers.Constant(value = delta),
                             trainable = True,
                             name = 'delta')

    def call(self, G, HH, yH, sigmat):
        # Start of new iteration
        taui_abs = KB.abs(self.taui) # no negative values for tau !
        if self.first_iter == True:
            ft = KB.softmax((KB.log(self.alpha) + G) * 1, axis = -1)
        else:
            ft = KB.softmax((KB.log(self.alpha) + G) * taui_abs, axis = -1)
        xt = KB.sum(ft * self.mt, axis = -1)
        xHH = KB.batch_dot(xt, HH)
        grad_x = taui_abs * (ft * self.mt - ft * KB.expand_dims(xt, axis = -1))
        # grad_L =  (1 - KB.exp(-G)) + 1 / sigmat ** 2 * grad_x * KB.expand_dims(xHH - yH, axis = -1) # original version
        grad_L = sigmat ** 2 * (1 - KB.exp(-G)) + grad_x * KB.expand_dims(xHH - yH, axis = -1)
        # Gradient/ResNet Layer
        G = G - self.delta * grad_L
        return G


class cmdnet_bin_kb(tf.keras.Model):
    """The binary CMDNet algorithm
    """
    def __init__(self, num_iter = 64, m = np.array([-1, 1]), alpha = np.array([0.5, 0.5]), taui0 = 1, delta0 = 1,
                    **kwargs):
        super(cmdnet_bin, self).__init__(**kwargs)
        self.it = tf.constant(num_iter)
        self.m = m
        self.M = m.shape[0]
        if (m == np.array([-1, 1])).all():
            alpha = alpha[:, 0]
        else:
            alpha = alpha[:, 1]
        self.alpha = tf.constant(value = alpha)
        taui0 = tf.constant(taui0, shape = (num_iter + 1), dtype = 'float32')
        delta0 = tf.constant(delta0, shape = (num_iter), dtype = 'float32')
        
        self.s0 = tf.constant(value = np.zeros_like(alpha))
        self.cmd_layer = []
        self.taui = []
        self.delta = []
        for ii in range(num_iter):
            if ii == 0:
                self.cmd_layer.append(cmdnet_bin_layer(m = self.m, alpha = self.alpha, taui = taui0[ii], delta = delta0[ii], first_iter = True))
            else:
                self.cmd_layer.append(cmdnet_bin_layer(m = self.m, alpha = self.alpha, taui = taui0[ii], delta = delta0[ii], first_iter = False))
            self.taui.append(self.cmd_layer[-1].taui)
            self.delta.append(self.cmd_layer[-1].delta)


    def call(self, inputs):
        [yt, Ht, sigmat0] = inputs
        HH = KB.batch_dot(KB.permute_dimensions(Ht, (0, 2, 1)), Ht)
        yH = KB.batch_dot(yt, Ht)
        s = KB.transpose(KB.expand_dims(self.s0)) * KB.ones_like(Ht[:, 0, :])
        sigmat = KB.expand_dims(sigmat0, axis = -1)
        for layer in self.cmd_layer:
            s = layer(s, HH, yH, sigmat)
        # Start of last iteration
        taui_abs = KB.abs(self.taui[-1]) # no negative values for tau !
        xt = KB.tanh((KB.log(1 / self.alpha - 1) + s) / 2 * taui_abs)
        xt2 = KB.expand_dims(xt, axis = -1)
        if (self.m == np.array([-1, 1])).all():
            ft = KB.concatenate([(1 - xt2) / 2, (1 + xt2) / 2], axis = -1) # [q(x = -1), q(x = 1)]
        else:
            ft = KB.concatenate([(1 + xt2) / 2, (1 - xt2) / 2], axis = -1) # [q(x = 1), q(x = -1)]
        return ft, xt

class cmdnet_bin_layer_kb(tf.keras.layers.Layer):
    """The binary CMDNet layer
    """
    def __init__(self, m = np.array([-1, 1]), alpha = np.array([0.5, 0.5]), taui = 1, delta = 1, first_iter = False):
        super().__init__()
        self.m = m
        self.alpha = alpha
        self.first_iter = first_iter
        self.taui = self.add_weight(shape = (1,),
                             initializer = tf.keras.initializers.Constant(value = taui),
                             trainable = True,
                             name = 'taui')
        self.delta = self.add_weight(shape = (1,),
                             initializer = tf.keras.initializers.Constant(value = delta),
                             trainable = True,
                             name = 'delta')

    def call(self, s, HH, yH, sigmat):
        # Start of new iteration
        taui_abs = KB.abs(self.taui) # no negative values for tau !
        if self.first_iter == True:
            xt = KB.tanh((KB.log(1 / self.alpha - 1) + s) / 2 * 1)
        else:
            xt = KB.tanh((KB.log(1 / self.alpha - 1) + s) / 2 * taui_abs)
        xt2 = KB.expand_dims(xt, axis = -1)
        if (self.m == np.array([-1, 1])).all():
            ft = KB.concatenate([(1 - xt2) / 2, (1 + xt2) / 2], axis = -1) # [q(x = -1), q(x = 1)]
        else:
            ft = KB.concatenate([(1 + xt2) / 2, (1 - xt2) / 2], axis = -1) # [q(x = 1), q(x = -1)]
        
        xHH = KB.batch_dot(xt, HH)
        grad_x = 1 / 2 * taui_abs * (1 - xt ** 2) 
        grad_L = sigmat ** 2 * KB.tanh(s / 2) + grad_x * (xHH - yH)
        # grad_L = KB.tanh(s / 2) + 1 / sigmat ** 2 * grad_x * (xHH - yH) # original version
        # Gradient/ResNet Layer
        s = s - self.delta * grad_L
        return s


class cmdnet1(tf.keras.Model):
    '''CMDNet layer
    '''
    def __init__(self, it, m, alpha, taui0 = 1, delta0 = 1):
        super(cmdnet1, self).__init__()
        self.it = tf.constant(it)
        self.alphat = tf.constant(value = alpha)
        self.m = tf.constant(value = m)
        self.M = m.shape[0]
        self.taui = self.add_weight(shape = (it + 1,),
                             initializer = tf.keras.initializers.Constant(value = taui0),
                             trainable = True)
        self.delta = self.add_weight(shape = (it,),
                             initializer = tf.keras.initializers.Constant(value = delta0),
                             trainable = True)
        self.G0 = tf.constant(value = np.zeros_like(alpha))
    
    # @tf.function
    def call(self, inputs):
        # inputs = [G, y, Hr, sigma]
        [yt, Ht, sigmat0] = inputs
        sigmat = tf.expand_dims(tf.expand_dims(sigmat0, axis = -1), axis = -1)
        
        G = KB.expand_dims(self.G0, axis = 0) * KB.expand_dims(KB.ones_like(Ht[:, 0, :]), axis = -1)
        taui_abs = KB.abs(self.taui[0])
        ft = KB.softmax((KB.log(self.alphat) + G) * taui_abs, axis = -1)
        mt = KB.expand_dims(KB.expand_dims(self.m, axis = 0), axis = 0)
        xt = KB.sum(ft * mt, axis = -1)
        
        # UNFOLDING
        HH = KB.batch_dot(KB.permute_dimensions(Ht, (0, 2, 1)), Ht)
        yH = KB.batch_dot(yt, Ht)
        for iteration in tf.range(0, self.it):
            xHH = KB.batch_dot(xt, HH)
            grad_x = taui_abs * (ft * mt- ft * KB.expand_dims(xt, axis = -1))
            # grad_L =  (1 - KB.exp(-G)) + 1 / sigmat ** 2 * grad_x * KB.expand_dims(xHH - yH, axis = -1) # original version
            grad_L = sigmat ** 2 * (1 - KB.exp(-G)) + grad_x * KB.expand_dims(xHH - yH, axis = -1)
            # Gradient/ResNet Layer
            G = G - self.delta[iteration] * grad_L
            # Start of new iteration
            taui_abs = KB.abs(self.taui[iteration + 1]) # no negative values for tau !
            ft = KB.softmax((KB.log(self.alphat) + G) * taui_abs, axis = -1)
            xt = KB.sum(ft * mt, axis = -1)
        return ft, xt


class cmdnet_bin1(tf.keras.Model):
    '''Binary CMDNet model
    '''
    def __init__(self, it, m, alpha, taui0 = 1, delta0 = 1):
        super(cmdnet_bin1, self).__init__()
        self.it = tf.constant(it)
        self.m = m
        self.M = m.shape[0]
        if (m == np.array([-1, 1])).all():
            alpha = alpha[:, 0]
        else:
            alpha = alpha[:, 1]
        self.alphat = tf.constant(value = alpha)
        self.taui = self.add_weight(shape = (it + 1,),
                             initializer = tf.keras.initializers.Constant(value = taui0),
                             trainable = True)
        self.delta = self.add_weight(shape = (it,),
                             initializer = tf.keras.initializers.Constant(value = delta0),
                             trainable = True)
        self.s0 = tf.constant(value = np.zeros_like(alpha))
    
    # @tf.function
    def call(self, inputs):
        [yt, Ht, sigmat0] = inputs
        sigmat = tf.expand_dims(sigmat0, axis = -1)
        
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
            taui_abs = KB.abs(self.taui[iteration + 1]) # no negative values for tau !
            xt = KB.tanh((KB.log(1 / alphat - 1) + s) / 2 * taui_abs)
            xt2 = KB.expand_dims(xt, axis = -1)
            if (self.m == np.array([-1, 1])).all():
                ft = KB.concatenate([(1 - xt2) / 2, (1 + xt2) / 2], axis = -1) # [q(x = -1), q(x = 1)]
            else:
                ft = KB.concatenate([(1 + xt2) / 2, (1 - xt2) / 2], axis = -1) # [q(x = 1), q(x = -1)]
        return ft, xt



if __name__ == '__main__':
#     my_func_main()
# def my_func_main():
    ber_plot = PlotBER()
    model1 = Model()

    # ber_plot.simulate(model1,
    #         np.arange(1, 12, 1), # -3, 16, 0.5
    #         batch_size = int(10000 / (model1.num_tx_ant * model1.num_tx_ant / 4)), # 4096
    #         max_mc_iter = 100, # 1000
    #         num_target_block_errors = 100,
    #         legend = "LMMSE (Uncorrelated)",
    #         show_fig = False);
    model1.algo = 0

    # Training
    # training parameters
    batch_size = 1
    train_iter = 50
    ebno_db_train = [7, 26]
    # clip_value_grad = 10 # gradient clipping for stable training convergence
    # delta0 = model1.trainable_variables[1]

    # bmi is used as metric to evaluate the intermediate results
    from sionna.utils.metrics import BitwiseMutualInformation
    bmi = BitwiseMutualInformation()

    # try also different optimizers or different hyperparameters
    optimizer = tf.keras.optimizers.Adam() # learning_rate = 1e-2) 

    for it in range(0, train_iter):
        with tf.GradientTape() as tape:
            b, llr, loss = model1(batch_size, ebno_db_train[0], ebno_db_train[1])     

        grads = tape.gradient(loss, model1.trainable_variables)
        # grads = tf.clip_by_value(grads, -clip_value_grad, clip_value_grad, name=None)
        optimizer.apply_gradients(zip(grads, model1.trainable_weights))

        # calculate and print intermediate metrics
        # only for information
        # this has no impact on the training
        if it % 10 == 0: # evaluate every 10 iterations
            # calculate ber from received LLRs
            b_hat = tf.cast(llr > 0, dtype = tf.float32) # hard decided LLRs first
            ber = compute_ber(b, b_hat)
            # and print results
            mi = bmi(b, llr).numpy() # calculate bit-wise mutual information
            l = loss.numpy() # copy loss to numpy for printing
            print(f"Current loss: {l:3f} ber: {ber:.4f} bmi: {mi:.3f}".format())
            bmi.reset_states() # reset the BMI metric


    
    ber_plot.simulate(model1,
            np.arange(1, 12, 1), # -3, 16, 0.5
            batch_size = int(10000 / (model1.num_tx_ant * model1.num_tx_ant / 4)), # 4096
            max_mc_iter = 100, # 1000
            num_target_block_errors = 100,
            legend = "CMDNet",
            # save_fig = True,
            show_fig = True);
    tplt.save("plots/MIMO_sionna_test.tikz")



    # DEBUGGEN: Erstmal cmdnet richtig programmieren!!!
    dtype0 = 'float32'
    batch_size = int(10000 / (model1.num_tx_ant * model1.num_tx_ant / 4)) # 4096
    b = model1.binary_source([batch_size, model1.num_tx_ant, model1.k])
    # c = model1.encoder(b)
    c = b
    x = model1.mapper(c)
    shape = tf.shape(x)
    x = tf.reshape(x, [-1, model1.num_tx_ant])
    ebno_db = tf.random.uniform(shape = [x.shape[0]], minval = 7, maxval = 26)
    no = 1 / 2 * ebnodb2no(ebno_db, model1.num_bits_per_symbol, 1) # model1.coderate)
    no_scale = model1.num_rx_ant * 2 / 2  # scaling of noise variance necessary since channel has Rayleigh taps with variance 1
    # corr_fac = 2
    # no *= corr_fac # * np.sqrt(model1.num_rx_ant)    # weil die rayleigh taps von h jeweils power 1 haben...
    y, h = model1.channel([x, no * no_scale])

    no_scale = model1.num_rx_ant * 2 / 2  # scaling of noise variance necessary since channel has Rayleigh taps with variance 1
    H_scale = np.sqrt(2 / model1.num_rx_ant)
    sigmat0 = tf.math.sqrt(no)
    x_hat, Phi_ee = mf.mmse([tf.math.real(y) * H_scale, tf.math.real(h) * H_scale, tf.repeat(sigmat0, y.shape[0])])
    no_eff = mf.tdiag2vec(Phi_ee)
    s = tf.complex(no * tf.eye(model1.num_rx_ant, model1.num_rx_ant), 0.0)
    x_hat2, no_eff2 = lmmse_equalizer(tf.cast(tf.math.real(y) * H_scale, dtype = 'complex64'), tf.cast(tf.math.real(h) * H_scale, dtype = 'complex64'), s)


    x_hat = tf.reshape(x_hat, shape)
    no_eff = tf.reshape(no_eff, shape)

    # # TODO: Demapper is not valid for probs!!!
    llr = model1.demapper([tf.cast(x_hat, dtype = 'complex64'), tf.cast(no_eff, dtype = 'float32')])
    # b_hat = model1.decoder(llr)

    # Wrapper
    # Ht = tf.concat([tf.concat([tf.math.real(h), tf.math.imag(h)], axis = -1), tf.concat([-tf.math.imag(h), tf.math.real(h)], axis = -1)], axis = -2) # / np.sqrt(model1.num_rx_ant)
    # yt = tf.concat([tf.math.real(y), tf.math.imag(y)], axis = -1)
    Ht = tf.math.real(h) * np.sqrt(2 / model1.num_rx_ant)
    yt = tf.math.real(y) * np.sqrt(2 / model1.num_rx_ant)
    sigmat0 = tf.math.sqrt(no)   # correct?
    # Complex ?
    # M = 2 ** model1.num_bits_per_symbol
    # Ht = h # tf.transpose(h, [0, 2, 1])
    # HHy = tf.squeeze(tf.matmul(tf.linalg.adjoint(Ht), tf.expand_dims(y, axis = -1)), axis = -1)
    # HHH = tf.matmul(tf.linalg.adjoint(Ht), Ht)
    # Equalization
    M = 2 # int(np.sqrt(2 ** model1.num_bits_per_symbol)) # we split real and imaginary part
    alpha = 1 / M * np.ones((model1.num_tx_ant, M), dtype = 'float32')
    # m = np.concatenate([tf.math.real(model1.mapper.constellation.points)[0::8].numpy(), tf.math.real(model1.mapper.constellation.points)[3::8].numpy()], axis = -1, dtype = 'float32')
    m = np.array([1, -1], dtype = 'float32')
    cmdnet0 = cmdnet_bin2(64, m, alpha)
    ft, xt = cmdnet0([yt, Ht, sigmat0])
    xtc = xt
    # xtc = tf.cast(xt[:, :int(xt.shape[-1] / 2)], dtype = 'complex64') + 1j * tf.cast(xt[:, int(xt.shape[-1] / 2):], dtype = 'complex64')
    # Get symbol indices for the transmitted symbols
    demapper_hard = SymbolDemapper("pam", model1.num_bits_per_symbol, hard_out=True)
    x_ind = demapper_hard([x, tf.ones(xtc.shape, dtype = 'float32')])
    # Get symbol indices for the received soft-symbols
    x_ind_hat = demapper_hard([tf.cast(xtc, dtype = 'complex64'), tf.ones(xtc.shape, dtype = 'float32')])
    Ps = compute_ser(x_ind, x_ind_hat)
    print(Ps)
    # LLR Demapping
    # wenn das klappt: alles zusammenfügen
    # TODO: object daraus machen mit konstanter maske
    # llr_c, p_c0 = tf_symprob2llr(ft, M)


