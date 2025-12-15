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
# from pickle import GLOBAL
import os
import time
import tikzplotlib as tplt
import numpy as np
import tensorflow as tf
# Import Sionna
import sionna as sn
from sionna.fec.ldpc.decoding import LDPC5GDecoder, LDPCBPDecoder
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.mapping import Mapper, Demapper, Constellation
from sionna.mimo import lmmse_equalizer
# For missing simulation of correlated massive MIMO channels
from sionna.channel.utils import exp_corr_mat
from sionna.channel import FlatFadingChannel, KroneckerModel
from sionna.utils import BinarySource, ebnodb2no, compute_ber, PlotBER
from sionna.utils.metrics import BitwiseMutualInformation
# import matplotlib.pyplot as plt
import cmdnet_utils_original as cmd_utils
import cmdnet_utils_tf2 as cmd_utils_tf2
import amp_layers
import cmdnet_layers

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Sionna only works with float32 precision in version 0.9.0
GLOBAL_PRECISION = 'float32'
print('Selected precision: ' + GLOBAL_PRECISION)
sn.config.xla_compat = True


class CommunicationModel(tf.keras.Model):
    '''
    Define the MIMO CommunicationModel
    '''

    def __init__(self, algo, spatial_corr=None, num_tx_ant=32, num_rx_ant=32, const=Constellation('pam', 1),
                 code=False, n=1024, k=512, code_it=10, code_train=False, trainbit=False, tf1_channel_code=False):
        super().__init__()
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant
        self.code = code
        self.trainbit = trainbit
        self.algo = algo  # choose algorithm, change in self.algo inactive with tf.function decorator
        self.binary_source = BinarySource()
        self.num_bits_per_symbol = const._num_bits_per_symbol
        self.mod = const._constellation_type
        # self.constellation = const # Constellation(mod, num_bits_per_symbol, trainable = False)
        self.mapper = Mapper(constellation=const)
        # self.demapper = Demapper("app", constellation = self.constellation)
        self.channel = FlatFadingChannel(self.num_tx_ant,
                                         self.num_rx_ant,
                                         spatial_corr=spatial_corr,
                                         add_awgn=True,
                                         return_channel=True)
        self.tf1_channel_code = tf1_channel_code
        # self.val_batch_size = 10000
        if self.code is True:
            if self.tf1_channel_code:
                # Original LDPC BP implementation from article
                file_code = np.load(os.path.join(
                    'codes', 'LDPC64x128') + '.npz')
                self.G = file_code['G']
                self.H = file_code['H']
                self.n = self.G.shape[1]
                self.k = self.G.shape[0]
                self.encoder = cmd_utils.LinearBlockEncoder(self.G)
                self.decoder = LDPCBPDecoder(
                    self.H, trainable=code_train, num_iter=code_it, cn_type='boxplus-phi', hard_out=False)
            else:
                self.n = n
                self.k = k
                self.encoder = LDPC5GEncoder(self.k, self.n)
                self.decoder = LDPC5GDecoder(self.encoder,
                                             num_iter=code_it,
                                             trainable=code_train,
                                             hard_out=False)
            self.coderate = self.k / self.n
        else:
            self.coderate = 1
    # Training of CMDNet w/o code works as expected if we remove this decorator (?)
    # @tf.function#(jit_compile = True)

    def call(self, batch_size, ebno_db, ebno_db_max=None):

        # Foward model
        if self.code:
            # Encoding: Horizontally coded MIMO as in Massive MIMO uplink
            # modulation_order = 2 ** (self.num_bits_per_symbol)
            b = self.binary_source(
                [batch_size * self.num_tx_ant * self.num_bits_per_symbol, self.k])
            b0 = tf.reshape(b, [batch_size, self.num_tx_ant,
                            self.num_bits_per_symbol, self.k])
            # c0 = cmd_utils.encoder(b0, self.G)    # Original LDPC BP implementation from article
            c0 = self.encoder(b0)
            # MIMO PAC encoding
            c = tf.reshape(tf.transpose(
                c0, [0, 3, 1, 2]), [-1, c0.shape[1], c0.shape[2]])
        else:
            b = self.binary_source([batch_size, self.num_tx_ant])
            c = b

        x = self.mapper(c)
        x = tf.reshape(x, [-1, self.num_tx_ant])

        if ebno_db_max is None:
            ebno_db_max = ebno_db
        ebno_db_vec = tf.random.uniform(
            shape=[tf.shape(x)[0]], minval=ebno_db, maxval=ebno_db_max)
        if self.mod == 'pam':
            no = 1 / 2 * ebnodb2no(ebno_db_vec,
                                   self.num_bits_per_symbol, self.coderate)
        else:
            no = ebnodb2no(
                ebno_db_vec, self.num_bits_per_symbol, self.coderate)
        # scaling of noise variance necessary since channel has Rayleigh taps with variance 1
        no_scale = self.num_rx_ant
        # no *= np.sqrt(self.num_rx_ant)    # original, why square root?
        y, h = self.channel([x, no * no_scale])
        sigmat0 = tf.math.sqrt(no)

        # Receiver side
        ft, x_hat, llr_c, p_c = self.algo([y, h, sigmat0])

        if self.code:
            # MIMO PAC decoding as in Massive MIMO uplink
            shapec = [-1, self.n, llr_c.shape[1], llr_c.shape[2]]
            llr_c2 = tf.reshape(tf.transpose(tf.reshape(llr_c, shapec), [
                                0, 2, 3, 1]), [-1, self.n])
            p_c = tf.reshape(tf.transpose(tf.reshape(
                p_c, shapec), [0, 2, 3, 1]), [-1, self.n])
        else:
            shapec = tf.shape(c)
            llr_c2 = tf.reshape(llr_c, shapec)
            p_c = tf.reshape(p_c, shapec)

        if self.code:
            if self.tf1_channel_code:
                # Original LDPC BP implementation from article
                # llr_hat, _, _ = cmd_utils.bp_decoder(llr_c2, self.H, 10, 0)
                # llr_b = llr_hat[..., 0:self.k]
                llr_hat = self.decoder(llr_c2)
                llr_b = llr_hat[..., 0:self.k]
            else:
                llr_b = self.decoder(llr_c2)
            symprob2llr = cmd_utils_tf2.TFSymprob2LLR(2, b=1)
            p_b = symprob2llr.llr2prob(llr_b)
        else:
            llr_b = llr_c2
            p_b = p_c
        b_hat = tf.cast(llr_b > 0, dtype=GLOBAL_PRECISION)

        # Training objective function
        if self.trainbit:
            bce = tf.keras.losses.BinaryCrossentropy()
            loss = bce(b, p_b)
        else:
            soft = 0
            if soft == 1:
                # 2. mean should be a sum for overall MSE scaling with Nt
                if not self.mod == 'pam':
                    x2 = tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)
                else:
                    x2 = tf.cast(x, dtype=GLOBAL_PRECISION)
                loss = tf.reduce_mean(
                    tf.reduce_mean((x2 - x_hat) ** 2, axis=-1))
            else:
                if self.algo.algo_name == 'MMSE':
                    loss = []
                else:
                    # 2. mean should be a sum since q factorizes
                    c_cl = tf.reshape(
                        c, [-1, self.num_tx_ant, self.num_bits_per_symbol])
                    cl = cmd_utils_tf2.tf_bin2int(c_cl, axis=-1)
                    if self.mod == 'pam':
                        cl = cmd_utils_tf2.tf_bin2int(c_cl, axis=-1)
                        M = 2 ** self.num_bits_per_symbol
                    else:
                        # e.g., c = 101010 -> c_re = 111, c_im = 000
                        c_cl_re = cmd_utils_tf2.tf_bin2int(
                            c_cl[..., ::2], axis=-1)
                        c_cl_im = cmd_utils_tf2.tf_bin2int(
                            c_cl[..., 1::2], axis=-1)
                        # concatenate classes of real and imaginary part as in equivalent real-valued system model
                        cl = tf.concat([c_cl_re, c_cl_im], axis=-1)
                        M = 2 ** int(self.num_bits_per_symbol / 2)
                    z = tf.one_hot(cl, depth=M)
                    loss = tf.reduce_mean(tf.reduce_mean(
                        tf.keras.losses.categorical_crossentropy(z, ft, axis=-1), axis=-1))
        return b, b_hat, loss, llr_c2, ft


def conventional_training(model, num_training_iterations, training_batch_size, ebno_db_train, it_print=100):
    '''
    Conventional training implementation of CMDNet
    '''
    # Optimizer used to apply gradients
    # try also different optimizers or different hyperparameters
    optimizer = tf.keras.optimizers.Adam()  # learning_rate = 1e-2)
    # optimizer = tf.keras.optimizers.SGD() # learning_rate = 1e-2)
    # clip_value_grad = 10 # gradient clipping for stable training convergence
    # bmi is used as metric to evaluate the intermediate results
    bmi = BitwiseMutualInformation()  # nur zur Auswertung

    # First iteration loss
    start_time = time.time()
    start_time2 = time.time()
    b, b_hat, loss, _, _ = model(training_batch_size,
                                 ebno_db_train[0], ebno_db_train[1])
    b = tf.cast(b, dtype=b_hat.dtype)
    ber = compute_ber(b, b_hat)
    if GLOBAL_PRECISION == 'float64':
        # TODO: Not working with tf.float64?
        mi = 0
    else:
        mi = bmi(b, b_hat).numpy()  # calculate bit-wise mutual information
    l = loss.numpy()  # copy loss to numpy for printing
    print(f"It: {0}/{num_training_iterations}, Train loss: {l:.6f}, BER: {ber:.4f}, BMI: {mi:.3f}, Time: {time.time() - start_time2:04.2f}s, Tot. time: ".format() +
          cmd_utils.print_time(time.time() - start_time))
    bmi.reset_states()  # reset the BMI metric
    start_time2 = time.time()

    @tf.function
    def train_step(optimizer, training_batch_size, ebno_db_train):
        # Forward pass
        with tf.GradientTape() as tape:
            b, b_hat, loss, _, _ = model(training_batch_size,
                                         ebno_db_train[0], ebno_db_train[1])
        # Computing and applying gradients
        grads = tape.gradient(loss, model.trainable_weights)
        # grads = tf.clip_by_value(grads, -clip_value_grad, clip_value_grad, name=None)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # The RX loss is returned to print the progress
        return b, b_hat, loss

    for it in range(num_training_iterations):
        b, b_hat, loss = train_step(
            optimizer, training_batch_size, ebno_db_train)
        # Printing periodically the training metrics
        if (it + 1) % it_print == 0:  # evaluate every it_print iterations
            b = tf.cast(b, dtype=b_hat.dtype)
            ber = compute_ber(b, b_hat)
            if GLOBAL_PRECISION == 'float64':
                # TODO: Not working with tf.float64?
                mi = 0
            else:
                # calculate bit-wise mutual information
                mi = bmi(b, b_hat).numpy()
            l = loss.numpy()  # copy loss to numpy for printing
            print(f"It: {it + 1}/{num_training_iterations}, Train loss: {l:.6f}, BER: {ber:.4f}, BMI: {mi:.3f}, Time: {time.time() - start_time2:04.2f}s, Tot. time: ".format(
            ) + cmd_utils.print_time(time.time() - start_time))
            bmi.reset_states()  # reset the BMI metric
            start_time2 = time.time()


def script1_cmdnet_symbol_detection_qpsk_32x32(max_mc_iter=1000, num_target_block_errors=100, batch_size=10000):
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
    constellation = Constellation(mod, num_bits_per_symbol, trainable=False)
    # Test params
    snr_range = np.arange(1, 18, 1)  # [1, 13, 1], [1, 31, 1], [-3, 16, 0.5]
    # snr_range = np.arange(1, 12, 1), # -3, 16, 0.5
    # int(10000 / (model3_cmdnet.num_tx_ant * model3_cmdnet.num_tx_ant / 4)), # 4096
    # batch_size = 10020
    # max_mc_iter = 10  # 1000
    # num_target_block_errors = 100

    # Load starting point
    algo0_mmse = cmd_utils_tf2.AlgoMMSE(constellation)
    algo1_amp = amp_layers.AlgoAMP(Nit, constellation, num_tx_ant)
    algo2_amp = amp_layers.AlgoAMP(Nit, constellation, num_tx_ant, binary=True)

    if GLOBAL_PRECISION == 'float64':
        sub_folder = 'data_cmdnet_sionna' + '_float64'
    else:
        sub_folder = 'data_cmdnet_sionna'
    # Old weight loading
    # saveobj2 = mf.savemodule('npz')
    # train_hist2 = mf.training_history()
    # sim_set = {'Mod': 'QPSK', 'Nr': 64, 'Nt': 64, 'L': 64,} # BPSK, QPSK, QAM16
    # fn = mf.filename_module('trainhist_', 'tf1_curves', 'CMD', '_binary_tau0.1', sim_set) # _binary_tau0.1, _tau0.1, _convex, _binary_tau0.075, '_binary_splin'
    # train_hist2.dict2obj(saveobj2.load(fn.pathfile))
    # [delta0, taui0] = train_hist2.params[-1]

    # , taui0 = taui0, delta0 = delta0)
    algo3_cmdnet = cmdnet_layers.AlgoCMDNet(
        Nit, constellation, num_tx_ant, binary=True)
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 *
               num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit, }
    fn = cmd_utils.filename_module(sub_folder, 'weights_',
                                   algo3_cmdnet.algo_name, 'binary_tau0.1', sim_set)
    algo3_cmdnet.load_weights(fn.pathfile)

    # , taui0 = taui0, delta0 = delta0)
    algo4_cmdnet_multiclass = cmdnet_layers.AlgoCMDNet(
        Nit, constellation, num_tx_ant, binary=False)
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 *
               num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit, }
    fn = cmd_utils.filename_module(sub_folder, 'weights_',
                                   algo4_cmdnet_multiclass.algo_name, 'tau0.1', sim_set)
    algo4_cmdnet_multiclass.load_weights(fn.pathfile)

    # , taui0 = taui0, delta0 = delta0
    algo5_cmdnet_NL16 = cmdnet_layers.AlgoCMDNet(
        16, constellation, num_tx_ant, binary=True)
    sim_set = {'Mod': mod + str(num_bits_per_symbol),
               'Nr': 2 * num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': 16, }
    fn = cmd_utils.filename_module(sub_folder, 'weights_',
                                   algo5_cmdnet_NL16.algo_name, 'binary_tau0.075', sim_set)
    algo5_cmdnet_NL16.load_weights(fn.pathfile)

    # , taui0 = taui0, delta0 = delta0
    algo6_cmdnet_splin = cmdnet_layers.AlgoCMDNet(
        Nit, constellation, num_tx_ant, binary=True)
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 *
               num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit, }
    fn = cmd_utils.filename_module(sub_folder, 'weights_',
                                   algo6_cmdnet_splin.algo_name, 'binary_splin', sim_set)
    algo6_cmdnet_splin.load_weights(fn.pathfile)

    delta0, taui0 = cmd_utils.CMD_initpar(
        M=2, L=64, typ='default', min_val=0.1)
    algo7_cmdnet_default_init = cmdnet_layers.AlgoCMDNet(Nit, constellation, num_tx_ant,
                                                         binary=True, taui0=taui0, delta0=delta0)
    delta0, taui0 = cmd_utils.CMD_initpar(
        M=2, L=64, typ='linear', min_val=0.01)
    algo8_cmdnet_splin_init = cmdnet_layers.AlgoCMDNet(Nit, constellation, num_tx_ant,
                                                       binary=True, taui0=taui0, delta0=delta0)

    # model1.algo.load_weights('data_cmdnet_sionna/weights')
    model0_mmse = CommunicationModel(algo=algo0_mmse, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant,
                                     const=constellation, code=code, trainbit=trainbit)
    model1_amp = CommunicationModel(algo=algo1_amp, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant,
                                    const=constellation, code=code, trainbit=trainbit)
    model2_amp = CommunicationModel(algo=algo2_amp, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant,
                                    const=constellation, code=code, trainbit=trainbit)
    model3_cmdnet = CommunicationModel(algo=algo3_cmdnet, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant,
                                       const=constellation, code=code, trainbit=trainbit)
    model4_cmdnet = CommunicationModel(algo=algo4_cmdnet_multiclass, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant,
                                       const=constellation, code=code, trainbit=trainbit)
    model5_cmdnet = CommunicationModel(algo=algo5_cmdnet_NL16, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant,
                                       const=constellation, code=code, trainbit=trainbit)
    model6_cmdnet = CommunicationModel(algo=algo6_cmdnet_splin, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant,
                                       const=constellation, code=code, trainbit=trainbit)
    model7_cmdnet = CommunicationModel(algo=algo7_cmdnet_default_init, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant,
                                       const=constellation, code=code, trainbit=trainbit)
    model8_cmdnet = CommunicationModel(algo=algo8_cmdnet_splin_init, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant,
                                       const=constellation, code=code, trainbit=trainbit)

    ber_plot.simulate(model0_mmse,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend="LMMSE (Uncorrelated)",
                      show_fig=False)
    ber_plot.simulate(model1_amp,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend="AMP N_it = 64",
                      show_fig=False)
    ber_plot.simulate(model2_amp,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend="AMP bin N_it = 64",
                      show_fig=False)
    ber_plot.simulate(model3_cmdnet,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend='CMDNet bin N_L = 64',
                      show_fig=False)
    ber_plot.simulate(model4_cmdnet,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend='CMDNet N_L = 64',
                      show_fig=False)
    ber_plot.simulate(model5_cmdnet,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend='CMDNet bin N_L = 16',
                      show_fig=False)
    ber_plot.simulate(model6_cmdnet,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend='CMDNet bin splin',
                      show_fig=False)
    ber_plot.simulate(model7_cmdnet,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend='CMDNet bin spdef Ntrain=0',
                      show_fig=False)
    ber_plot.simulate(model8_cmdnet,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend='CMDNet bin splin Ntrain=0',
                      # save_fig = True,
                      show_fig=False)

    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 * num_rx_ant,
               'Nt': 2 * num_tx_ant, 'L': Nit}
    curve_file = plot_cmdnet_tf1_curve(sim_set)
    # Other original TF1 simulation results curves
    # curve_file = plot_cmdnet_tf1_curve(sim_set, fn_extension='binary_splin')
    # curve_file = plot_cmdnet_tf1_curve(sim_set, fn_extension='tau0.1')
    # sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 * num_rx_ant,
    #            'Nt': 2 * num_tx_ant, 'L': 16}
    # curve_file = plot_cmdnet_tf1_curve(sim_set)
    ber_plot(snr_db=curve_file['ebn0'],
             ber=curve_file['ber'], legend='CMDNet TF1', is_bler=False)
    try:
        tplt.save('plots/cmdnet_sionna_script1_QAM4_64x64.tikz')
    except Exception as e:
        print(e)

    return ber_plot


def plot_cmdnet_tf1_curve(sim_set, fn_extension='', algo='CMD'):
    '''Plots the curve of the original CMDNet implementation in TensorFlow 1
    Reference curve for new TF2 implementation
    mod_curves BPSK, QPSK, QAM16
    '''
    sim_set_converted = sim_set
    sim_set_converted['Mod'] = sionna_modulation2cmdnet_modulation_name(
        sim_set_converted['Mod'])
    mod_curves = sim_set_converted['Mod']
    nrx = sim_set_converted['Nr']
    ntx = sim_set_converted['Nt']
    nit = sim_set_converted['L']

    # Select specific CMDNet curve from TensorFlow 1 implementation

    if fn_extension == '':
        if mod_curves == 'QAM16' and ntx == 64 and nrx == 64 and nit == 64:
            fn_extension = 'convex'
        elif mod_curves == 'QPSK' and ntx == 64 and nrx == 64 and nit == 64:
            fn_extension = 'binary_tau0.1'
            # fn_extension = 'binary_splin'
            # fn_extension = 'tau0.1'
        elif mod_curves == 'QPSK' and ntx == 64 and nrx == 64 and nit == 16:
            fn_extension = 'binary_tau0.075'
        elif mod_curves == 'QPSK' and ntx == 16 and nrx == 16 and nit == 16:
            fn_extension = 'binary'
        else:
            print('No specific curve selected.')
            fn_extension = ''

    fn = cmd_utils.filename_module(
        'tf1_curves', 'RES_', algo, fn_extension, sim_set_converted, code_set=0, tf=1)

    pathfile = fn.pathfile + '.npz'
    curve_file = np.load(pathfile)
    # plt.semilogy(curve_file['ebn0'], curve_file['ber'],
    #              'r-o', label='CMDNet TF1')
    return curve_file


def sionna_modulation2cmdnet_modulation_name(mod):
    '''Translates Sionna name of modulations to CMDNet standard
    '''
    if mod == 'qam2':
        mod_curves = 'QPSK'
    elif mod == 'qam4':
        mod_curves = 'QAM16'
    else:
        print('Not available in curves.')
        mod_curves = mod
    return mod_curves


def script2_cmdnet_symbol_detection_qpsk_8x8(max_mc_iter=1000, num_target_block_errors=100, batch_size=10000):
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
    constellation = Constellation(mod, num_bits_per_symbol, trainable=False)
    # Test params
    snr_range = np.arange(1, 16, 1)  # [1, 13, 1], [1, 31, 1], [-3, 16, 0.5]
    # snr_range = np.arange(1, 12, 1), # -3, 16, 0.5
    # int(10000 / (model1.num_tx_ant * model1.num_tx_ant / 4)), # 4096
    # batch_size = 10020
    # max_mc_iter = 100  # 1000
    # num_target_block_errors = 100

    # Load starting point
    algo0_mmse = cmd_utils_tf2.AlgoMMSE(constellation)
    # algo1_amp = amp_layers.AlgoAMP(Nit, constellation, num_tx_ant)
    algo2_amp = amp_layers.AlgoAMP(Nit, constellation, num_tx_ant, binary=True)

    if GLOBAL_PRECISION == 'float64':
        sub_folder = 'data_cmdnet_sionna' + '_float64'
    else:
        sub_folder = 'data_cmdnet_sionna'

    # , taui0 = taui0, delta0 = delta0)
    algo3_cmdnet = cmdnet_layers.AlgoCMDNet(
        Nit, constellation, num_tx_ant, binary=True)
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 *
               num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit, }
    fn = cmd_utils.filename_module(sub_folder, 'weights_',
                                   algo3_cmdnet.algo_name, 'binary', sim_set)
    algo3_cmdnet.load_weights(fn.pathfile)

    model0_mmse = CommunicationModel(algo=algo0_mmse, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant,
                                     const=constellation, code=code, trainbit=trainbit)
    # model1_amp = CommunicationModel(algo = algo1_amp, num_tx_ant = num_tx_ant, num_rx_ant = num_rx_ant, const = constellation, code = code, trainbit = trainbit)
    model2_amp = CommunicationModel(algo=algo2_amp, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant,
                                    const=constellation, code=code, trainbit=trainbit)
    model3_cmdnet = CommunicationModel(algo=algo3_cmdnet, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant,
                                       const=constellation, code=code, trainbit=trainbit)

    ber_plot.simulate(model0_mmse,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend="LMMSE (Uncorrelated)",
                      show_fig=False)
    # ber_plot.simulate(model1_amp,
    #         snr_range,
    #         batch_size = batch_size,
    #         max_mc_iter = max_mc_iter,
    #         num_target_block_errors = num_target_block_errors,
    #         legend = "AMP N_it = 16",
    #         show_fig = False);
    ber_plot.simulate(model2_amp,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend="AMP bin N_it = 16",
                      show_fig=False)
    ber_plot.simulate(model3_cmdnet,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend='CMDNet bin N_L = 16',
                      # save_fig = True,
                      show_fig=False)

    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 * num_rx_ant,
               'Nt': 2 * num_tx_ant, 'L': Nit}
    curve_file = plot_cmdnet_tf1_curve(sim_set)
    ber_plot(snr_db=curve_file['ebn0'],
             ber=curve_file['ber'], legend='CMDNet TF1', is_bler=False)

    try:
        tplt.save('plots/cmdnet_sionna_script2_QAM4_16x16.tikz')
    except Exception as e:
        print(e)
    return ber_plot


def script3_cmdnet_symbol_detection_qam16_32x32(max_mc_iter=1000, num_target_block_errors=1000, batch_size=10000):
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
    constellation = Constellation(mod, num_bits_per_symbol, trainable=False)
    # Test params
    snr_range = np.arange(6, 31, 1)
    # batch_size = 10020
    # max_mc_iter = 100  # 1000
    # num_target_block_errors = 1000

    # Load starting point
    algo0_mmse = cmd_utils_tf2.AlgoMMSE(constellation)
    algo1_amp = amp_layers.AlgoAMP(Nit, constellation, num_tx_ant)

    if GLOBAL_PRECISION == 'float64':
        sub_folder = 'data_cmdnet_sionna' + '_float64'
    else:
        sub_folder = 'data_cmdnet_sionna'

    # , taui0 = taui0, delta0 = delta0)
    algo2_cmdnet = cmdnet_layers.AlgoCMDNet(Nit, constellation, num_tx_ant)
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 *
               num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit, }
    fn = cmd_utils.filename_module(sub_folder, 'weights_',
                                   algo2_cmdnet.algo_name, 'convex', sim_set)
    algo2_cmdnet.load_weights(fn.pathfile)

    model0_mmse = CommunicationModel(algo=algo0_mmse, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant,
                                     const=constellation, code=code, trainbit=trainbit)
    model1_amp = CommunicationModel(algo=algo1_amp, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant,
                                    const=constellation, code=code, trainbit=trainbit)
    model2_cmdnet = CommunicationModel(algo=algo2_cmdnet, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant,
                                       const=constellation, code=code, trainbit=trainbit)

    ber_plot.simulate(model0_mmse,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend="LMMSE (Uncorrelated)",
                      show_fig=False)
    ber_plot.simulate(model1_amp,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend="AMP N_it = 64",
                      show_fig=False)
    ber_plot.simulate(model2_cmdnet,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend='CMDNet N_L = 64',
                      # save_fig = True,
                      show_fig=False)

    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 * num_rx_ant,
               'Nt': 2 * num_tx_ant, 'L': Nit}
    curve_file = plot_cmdnet_tf1_curve(sim_set)
    ber_plot(snr_db=curve_file['ebn0'],
             ber=curve_file['ber'], legend='CMDNet TF1', is_bler=False)

    try:
        tplt.save('plots/cmdnet_sionna_script3_QAM16_64x64.tikz')
    except Exception as e:
        print(e)
    return ber_plot


def script4_cmdnet_qpsk_32x32_with_channel_code(max_mc_iter=100, num_target_block_errors=100, batch_size=10, tf1_channel_code=False):
    '''All CMDNet curves with channel code from the journal article for QPSK modulation and dimension 32x32 (effective BPSK modulation and dimension: 64x64)
    Idea: Try using with old BP implementation -> no change -> problem lies within new cmdnet implementation
    float64 precision required for reproduction, worse results with float32 -> not implementable in Sionna
    '''
    # tf.config.run_functions_eagerly(True)

    ber_plot = PlotBER()
    mod = 'qam'
    num_tx_ant = 32
    num_rx_ant = 32
    Nit = 2 * num_tx_ant    # 64
    num_bits_per_symbol = 2
    constellation = Constellation(mod, num_bits_per_symbol, trainable=False)
    n = 128     # 128, 1024
    k = 64      # 64, 512
    Ncit = 10
    # Test params
    snr_range = np.arange(3, 14, 1)     # np.arange(3, 19, 1)
    # batch_size = 10
    # max_mc_iter = 100  # 100
    # num_target_block_errors = 100  # 1000

    # Load starting point
    algo0_mmse = cmd_utils_tf2.AlgoMMSE(constellation)
    algo1_amp = amp_layers.AlgoAMP(Nit, constellation, num_tx_ant)

    if GLOBAL_PRECISION == 'float64':
        sub_folder = 'data_cmdnet_sionna' + '_float64'
    else:
        sub_folder = 'data_cmdnet_sionna'

    algo2_cmdnet = cmdnet_layers.AlgoCMDNet(
        Nit, constellation, num_tx_ant, binary=True)
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 *
               num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit, }
    fn = cmd_utils.filename_module(sub_folder, 'weights_',
                                   algo2_cmdnet.algo_name, 'binary_tau0.1', sim_set)
    algo2_cmdnet.load_weights(fn.pathfile)

    model0_mmse = CommunicationModel(algo=algo0_mmse, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant, const=constellation,
                                     code=True, n=n, k=k, code_it=Ncit, tf1_channel_code=tf1_channel_code)
    model1_amp = CommunicationModel(algo=algo1_amp, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant, const=constellation,
                                    code=True, n=n, k=k, code_it=Ncit, tf1_channel_code=tf1_channel_code)
    model2_cmdnet_with_code = CommunicationModel(algo=algo2_cmdnet, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant, const=constellation,
                                                 code=True, n=n, k=k, code_it=Ncit, tf1_channel_code=tf1_channel_code)
    model3_cmdnet = CommunicationModel(algo=algo2_cmdnet, num_tx_ant=num_tx_ant,
                                       num_rx_ant=num_rx_ant, const=constellation)

    ber_plot.simulate(model0_mmse,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend="LMMSE (Uncorrelated)",
                      #   add_ber=False,
                      #   add_bler=True,
                      add_ber=False,
                      add_bler=True,
                      show_fig=False)
    ber_plot.simulate(model1_amp,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend="AMP bin N_it = 64",
                      add_ber=False,
                      add_bler=True,
                      show_fig=False)
    ber_plot.simulate(model2_cmdnet_with_code,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend='CMDNet bin N_L = 64',
                      add_ber=False,
                      add_bler=True,
                      show_fig=False)
    ber_plot.simulate(model3_cmdnet,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend='CMDNet bin N_L = 64 uncoded',
                      add_ber=False,
                      add_bler=True,
                      # save_fig = True,
                      show_fig=False)

    if tf1_channel_code is True:
        sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 * num_rx_ant,
                   'Nt': 2 * num_tx_ant, 'L': Nit}
        curve_file = plot_cmdnet_tf1_curve(
            sim_set, fn_extension='LDPC64x128bphoriz_binary_tau0.1_float64')
        curve_file2 = plot_cmdnet_tf1_curve(
            sim_set, fn_extension='LDPC64x128bphoriz_binary_tau0.1')
        curve_file3 = plot_cmdnet_tf1_curve(
            sim_set, fn_extension='LDPC64x128bphoriz', algo='AMP')
        curve_file4 = plot_cmdnet_tf1_curve(
            sim_set, fn_extension='LDPC64x128bphoriz', algo='MMSE')
        curve_file5 = plot_cmdnet_tf1_curve(sim_set)
        # ber_plot(snr_db=[curve_file['cebn0'], curve_file2['cebn0']],
        #          ber=[curve_file['cfer'], curve_file2['cfer']], legend=['CMDNet + LDPC64x128 TF1 float64', 'CMDNet + LDPC64x128 TF1 float32'], is_bler=[True, True])
        ber_plot(snr_db=[curve_file['cebn0'], curve_file2['cebn0'], curve_file3['cebn0'], curve_file4['cebn0'], curve_file5['cebn0']],
                 ber=[curve_file['cfer'], curve_file2['cfer'], curve_file3['cfer'], curve_file4['cfer'], curve_file5['fer']], legend=['CMDNet + LDPC64x128 TF1 float64', 'CMDNet + LDPC64x128 TF1 float32', 'AMP + LDPC64x128 TF1 float64', 'MMSE + LDPC64x128 TF1 float64', 'CMDNet bin N_L = 64 uncoded TF1'], is_bler=[True, True, True, True, True])
    try:
        tplt.save('plots/cmdnet_sionna_script4_QAM4_64x64_wcode.tikz')
    except Exception as e:
        print(e)
    return ber_plot


def script5_cmdnet_qpsk_32x32_training(max_mc_iter=100, num_target_block_errors=100, batch_size=10000, it_print=100, code=0, tf1_channel_code=False, load_pretrained_cmdnet=False, trainbit=False):
    '''Exemplary training for binary CMDNet for QPSK modulation and dimension 32x32 (effective BPSK modulation and dimension: 64x64)
    code: Joint training with code - New simulation/idea
    '''
    # tf.config.run_functions_eagerly(True)

    # Training
    mod = 'qam'
    num_tx_ant = 32
    num_rx_ant = 32
    Nit = num_tx_ant * 2
    num_bits_per_symbol = 2
    constellation = Constellation(mod, num_bits_per_symbol, trainable=False)
    code_it = 10
    code_train = False
    # Test params
    ber_plot = PlotBER()
    snr_range = np.arange(1, 18, 1)
    # batch_size = 10020
    # max_mc_iter = 100                   # 100
    # num_target_block_errors = 100       # 100
    # it_print = 1

    # Algorithm and Model
    if GLOBAL_PRECISION == 'float64':
        sub_folder = 'data_cmdnet_sionna' + '_float64'
    else:
        sub_folder = 'data_cmdnet_sionna'

    # Original initialization of CMDNet
    delta0, taui0 = cmd_utils.CMD_initpar(
        M=2, L=Nit, typ='default', min_val=0.1)
    algo1_cmdnet = cmdnet_layers.AlgoCMDNet(Nit, constellation, num_tx_ant,
                                            binary=True, taui0=taui0, delta0=delta0)
    # Compared to final trained weights from TF1
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 *
               num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit, }
    fn = cmd_utils.filename_module(sub_folder, 'weights_',
                                   algo1_cmdnet.algo_name, 'binary_tau0.1', sim_set)
    if load_pretrained_cmdnet is True:
        # Load pretrained weights
        algo1_cmdnet.load_weights(fn.pathfile)

    model1_cmdnet = CommunicationModel(algo=algo1_cmdnet, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant,
                                       const=constellation, code=code, code_it=code_it, code_train=code_train, tf1_channel_code=tf1_channel_code, trainbit=trainbit)

    # Training parameters
    train_iter = 100000                 # 100000
    # w/o code: 1000 -> 500 ?, w code: 10/1
    if code is True:
        training_batch_size = 10
    else:
        training_batch_size = 500
    # w/o code, QPSK: [7, 26], QAM16: [10, 33], w code: [0, 3]
    if code is True:
        ebno_db_train = [0, 3]
    else:
        ebno_db_train = [4, 27]
    conventional_training(model1_cmdnet, train_iter, training_batch_size,
                          ebno_db_train, it_print=it_print)
    # Save after training
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 *
               num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit, }
    fn = cmd_utils.filename_module(sub_folder, 'weights_',
                                   algo1_cmdnet.algo_name, 'test', sim_set)
    # Path has to be shorter...
    path = os.path.join(fn.path, 'weights')
    model1_cmdnet.algo.save_weights(path)

    # Comparison to final trained weights from TF1
    algo2_cmdnet_trained = cmdnet_layers.AlgoCMDNet(Nit, constellation, num_tx_ant,
                                                    binary=True, taui0=taui0, delta0=delta0)
    algo2_cmdnet_trained.load_weights(fn.pathfile)
    model2_cmdnet_trained = CommunicationModel(algo=algo2_cmdnet_trained, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant,
                                               const=constellation, code=code, code_it=code_it, tf1_channel_code=tf1_channel_code, trainbit=trainbit)

    ber_plot.simulate(model1_cmdnet,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend='CMDNet bin TF2 trained',
                      show_fig=False)
    ber_plot.simulate(model2_cmdnet_trained,
                      snr_range,
                      batch_size=batch_size,
                      max_mc_iter=max_mc_iter,
                      num_target_block_errors=num_target_block_errors,
                      legend='CMDNet bin N_L = 64',
                      show_fig=False)

    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 * num_rx_ant,
               'Nt': 2 * num_tx_ant, 'L': Nit}
    curve_file = plot_cmdnet_tf1_curve(sim_set)
    ber_plot(snr_db=curve_file['ebn0'],
             ber=curve_file['ber'], legend='CMDNet TF1', is_bler=False)

    try:
        tplt.save('plots/cmdnet_sionna_script5_QAM4_64x64_training.tikz')
    except Exception as e:
        print(e)
    return ber_plot


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():
    # tf.debugging.enable_check_numerics()
    # tf.config.run_functions_eagerly(True)
    cmd_utils.gpu_select(number=0, memory_growth=True, cpus=0)
    tf.keras.backend.set_floatx(GLOBAL_PRECISION)

    # Choose example test script
    # 0: CMDNet QPSK 32x32 (BPSK 64x64)
    # 1: CMDNet QPSK 8x8 (BPSK 16x16)
    # 2: CMDNet QAM16 32x32 (QAM4 64x64)
    # 3: CMDNet QPSK 32x32 (BPSK 64x64) with subsequent 128x64/new 5G channel code
    # 4: Training of CMDNet QPSK 32x32 (BPSK 64x64) (Also works with trainbit = True)
    # 5: New joint training of CMDNet and 128x64/5G channel coding, QPSK 32x32 (BPSK 64x64)
    # 5 is so far not numerically and needs debugging (NaN at training time)

    EXAMPLE = 5
    # Use 128x64 from journal code (TF1_CHANNEL_CODE = True) or 5G channel code (TF1_CHANNEL_CODE = False)
    TF1_CHANNEL_CODE = False
    # Simulation parameters defining plot accuracy
    MAX_MC_ITER = 100              # 1000 in article, 100 for faster simulations
    NUM_TARGET_BLOCK_ERRORS = 100  # 1000 in article, 100 for faster simulations

    if EXAMPLE == 0:
        ber_plot = script1_cmdnet_symbol_detection_qpsk_32x32(max_mc_iter=MAX_MC_ITER,
                                                              num_target_block_errors=NUM_TARGET_BLOCK_ERRORS)
    elif EXAMPLE == 1:
        ber_plot = script2_cmdnet_symbol_detection_qpsk_8x8(max_mc_iter=MAX_MC_ITER,
                                                            num_target_block_errors=NUM_TARGET_BLOCK_ERRORS)
    elif EXAMPLE == 2:
        ber_plot = script3_cmdnet_symbol_detection_qam16_32x32(max_mc_iter=MAX_MC_ITER,
                                                               num_target_block_errors=NUM_TARGET_BLOCK_ERRORS)
    elif EXAMPLE == 3:
        ber_plot = script4_cmdnet_qpsk_32x32_with_channel_code(max_mc_iter=100,
                                                               num_target_block_errors=NUM_TARGET_BLOCK_ERRORS, batch_size=10, tf1_channel_code=TF1_CHANNEL_CODE)
    elif EXAMPLE == 4:
        ber_plot = script5_cmdnet_qpsk_32x32_training(max_mc_iter=MAX_MC_ITER,
                                                      num_target_block_errors=NUM_TARGET_BLOCK_ERRORS, it_print=100, trainbit=False)
    elif EXAMPLE == 5:
        ber_plot = script5_cmdnet_qpsk_32x32_training(max_mc_iter=MAX_MC_ITER,
                                                      num_target_block_errors=NUM_TARGET_BLOCK_ERRORS, it_print=100, load_pretrained_cmdnet=True, code=True, trainbit=True, tf1_channel_code=TF1_CHANNEL_CODE)
    else:
        print('No test script selected.')
