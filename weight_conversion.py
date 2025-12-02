#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue July 12 11:32:57 2022

@author: beck
"""
import sys                                  # NOQA
# Include current folder, where start simulation script and packages are
sys.path.append('.')                        # NOQA
# Include parent folder, where own packages are
sys.path.append('..')                       # NOQA

import tensorflow as tf
from sionna.mapping import Constellation
import my_training as mt
import my_functions as mf
import cmdnet_sionna_tf2 as cmd_sionna
import cmdnet_utils as cmd_utils

GLOBAL_PRECISION = 'float64'


def weight_conversion():
    '''Weight saving: Conversion of old weight data into new separate files
    '''
    mod = 'qam'
    mod0 = 'QAM16'  # BPSK, QPSK, QAM16
    # _binary_tau0.1, _binary, _tau0.1, _convex, _binary_tau0.075, _binary_splin
    fn_ext = 'convex'
    num_tx_ant = 32
    num_rx_ant = 32
    num_bits_per_symbol = 4
    Nit = 64
    binary = False
    constellation = Constellation(mod, num_bits_per_symbol, trainable=False)

    saveobj2 = mf.savemodule('npz')
    train_hist2 = mt.TrainingHistory()
    sim_set = {'Mod': mod0, 'Nr': 2 * num_rx_ant,
               'Nt': 2 * num_tx_ant, 'L': Nit, }
    fn = mf.filename_module('trainhist_', 'curves',
                            'CMD', '_' + fn_ext, sim_set)
    train_hist2.dict2obj(saveobj2.load(fn.pathfile))
    [delta0, taui0] = train_hist2.params[-1]
    # delta0, taui0 = CMD_initpar(M = 2, L = 64, typ = 'default', min_val = 0.1)

    algo1 = cmd_sionna.algo_cmdnet(Nit, constellation, num_tx_ant,
                                   binary=binary, taui0=taui0, delta0=delta0)
    # algo1 = algo_cmdnet(Nit, constellation, num_tx_ant, binary = binary, taui0 = taui0, delta0 = delta0)
    sim_set = {'Mod': mod + str(num_bits_per_symbol), 'Nr': 2 *
               num_rx_ant, 'Nt':  2 * num_tx_ant, 'L': Nit, }
    if GLOBAL_PRECISION == 'float64':
        sub_folder = 'data_MIMO_sionna' + '_float64'
    else:
        sub_folder = 'data_MIMO_sionna'
    fn2 = cmd_utils.filename_module(
        sub_folder, 'weights_', algo1.algo_name, fn_ext, sim_set)
    # fn2 = filename_module('data_MIMO_sionna', 'weights_', algo1.algo_name, fn_ext, sim_set)
    algo1.save_weights(fn2.pathfile)
    # algo1.load_weights(fn2.pathfile)

    return delta0, taui0


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():
    tf.keras.backend.set_floatx(GLOBAL_PRECISION)
    delta0, taui0 = weight_conversion()
