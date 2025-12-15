# CMDNet for Sionna and TensorFlow 2: Concrete MAP Detection Network
CMDNet implementation for Soft MIMO Detection in Nvidia's Sionna library and TensorFlow 2.

This new CMDNet version is adapted for TensorFlow 2, as the original source code [1] has been implemented in TensorFlow 1.15. Furthermore, we implemented CMDNet for Sionna using Keras as an agnostic backend to make it widely applicable for communication engineers. This repository also includes the Approximate Message Passing (AMP) algorithm.

You can find the original source code [1] from the scientific research articles [2, 3] on [GitHub](https://github.com/ant-uni-bremen/CMDNet) and on [Zenodo](https://doi.org/10.5281/zenodo.8416507):

1. Edgar Beck, Concrete MAP Detection Network (CMDNet) Software, Zenodo, version v1.0.2, Oct. 2023. doi: 10.5281/zenodo.8416507.
2. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “CMDNet: Learning a Probabilistic Relaxation of Discrete Variables for Soft Detection With Low Complexity,” IEEE Trans. Commun., vol. 69, no. 12, pp. 8214–8227, Dec. 2021. https://doi.org/10.1109/TCOMM.2021.3114682
3. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Concrete MAP Detection: A Machine Learning Inspired Relaxation,” in 24th International ITG Workshop on Smart Antennas (WSA 2020), vol. 24, Hamburg, Germany, Feb. 2020, pp. 1–5.

# Usage

Choose an example test script in `cmdnet_sionna_tf2.py` by setting the variables, e.g., `EXAMPLE = 0`:

### Scenarios
0. CMDNet QPSK 32x32
1. CMDNet QPSK 8x8
2. CMDNet QAM16 32x32
3. CMDNet QPSK 32x32 with code
4. Training of CMDNet QPSK 32x32
5. New joint training of CMDNet with code (so far numerically unstable, debugging needed...)

All journal article experiments such as CMDNet symbol detection are reproducible. Correlated massive MIMO channel matrices have not been implemented so far.

However, CMDNet and LMMSE with the 128x64 LDPC channel code perform better in the TensorFlow 2 Sionna implementation. Further research is required to clarify where the reason exactly lies. Note that CMDNet without channel coding still outperforms LMMSE with channel coding.


# Requirements

We implemented CMDNet for
- Sionna 0.9.0
- TensorFlow 2.6.0
- The Conda environment can be found in `sionna.yml`.

Changes may be required for the newest Sionna version, but the code is Keras-native and adaptation may be straightforward.

# Acknowledgements

This work was partly funded by the German Ministry of Education and Research (BMBF) under grant 16KIS1028 (MOMENTUM).

# License and Referencing

This program is licensed under the GPLv3 license. If you in any way use this code for research that results in publications, please cite our original article listed below.


# Abstract of the articles

2. Following the great success of Machine Learning (ML), especially Deep Neural Networks (DNNs), in many research domains in 2010s, several ML-based approaches were proposed for detection in large inverse linear problems, e.g., massive MIMO systems. The main motivation behind is that the complexity of Maximum A-Posteriori (MAP) detection grows exponentially with system dimensions. Instead of using DNNs, essentially being a black-box, we take a slightly different approach and introduce a probabilistic Continuous relaxation of disCrete variables to MAP detection. Enabling close approximation and continuous optimization, we derive an iterative detection algorithm: Concrete MAP Detection (CMD). Furthermore, extending CMD by the idea of deep unfolding into CMDNet, we allow for (online) optimization of a small number of parameters to different working points while limiting complexity. In contrast to recent DNN-based approaches, we select the optimization criterion and output of CMDNet based on information theory and are thus able to learn approximate probabilities of the individual optimal detector. This is crucial for soft decoding in today’s communication systems. Numerical simulation results in MIMO systems reveal CMDNet to feature a promising accuracy complexity trade-off compared to State of the Art. Notably, we demonstrate CMDNet’s soft outputs to be reliable for decoders.

3. Motivated by large linear inverse problems where the complexity of the Maximum A-Posteriori (MAP) detector grows exponentially with system dimensions, e.g., large MIMO, we introduce a method to relax a discrete MAP problem into a continuous one. The relaxation is inspired by recent ML research and offers many favorable properties reflecting its quality. Hereby, we derive an iterative detection algorithm based on gradient descent optimization: Concrete MAP Detection (CMD). We show numerical results of application in large MIMO systems that demonstrate superior performance w.r.t. all considered State of the Art approaches.