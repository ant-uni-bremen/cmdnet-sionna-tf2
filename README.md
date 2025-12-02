# CMDNet_Sionna_tf2
CMDNet implementation for Soft MIMO Detection in Nvidia's Sionna library and TensorFlow 2.

This new CMDNet version is adapted for TensorFlow 2, as the original source code has been implemented in TensorFlow 1.15. Furthermore, we implemented CMDNet for Sionna using Keras as an agnostic backend to make it widely applicable for communication engineers. It also includes the Approximate Message Passing (AMP) algorithm.

You can find the original source code [here](https://github.com/ant-uni-bremen/CMDNet) and on [Zenodo](https://doi.org/10.5281/zenodo.8416507).

# Usage

Choose an example test script in `cmdnet_sionna_tf2.py` by setting variable, e.g., `example = 0`:

### Scenarios
0. CMDNet QPSK 64x64
1. CMDNet QPSK 16x16
2. CMDNet QAM16 64x64
3. CMDNet QPSK 64x64 with code (float32 precision problem...)
4. Training of CMDNet QPSK 64x64

### Disclaimer
Sionna and TensorFlow 2 make it difficult to use float64 computation accuracy for training and inference, which we used in the TensorFlow 1 version experiments. However, numerical experiments in TensorFlow 1 show that joint CMDNet and decoder performance decreases significantly with float32 instead of float64 computation accuracy. This explains why this TensorFlow 2 version cannot reproduce the joint equalization and decoding results in Scenario 3. We note that only specific changes may be required to make it work, not necessarily switching to overall float64 precision. Further research is required to clarify where the reason exactly lies.

All other experiments such as CMDNet symbol detection are reproducible.

# Requirements

We implemented CMDNet for
- Sionna 0.9.0
- TensorFlow 2.6.0
- Copy dependencies of conda environment...

Changes may be required for the newest Sionna version, but the code is Keras-native and adaptation may be straightforward.

# Acknowledgements

This work was partly funded by the German Ministry of Education and Research (BMBF) under grant 16KIS1028 (MOMENTUM).

# License and Referencing

This program is licensed under the GPLv3 license. If you in any way use this code for research that results in publications, please cite our original article listed below.

# Publications

Updated version of source code of
1. Edgar Beck, Concrete MAP Detection Network (CMDNet) Software, Zenodo, version v1.0.2, Oct. 2023. doi: 10.5281/zenodo.8416507.

from scientific research articles [2, 3]:

2. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “CMDNet: Learning a Probabilistic Relaxation of Discrete Variables for Soft Detection With Low Complexity,” IEEE Trans. Commun., vol. 69, no. 12, pp. 8214–8227, Dec. 2021. https://doi.org/10.1109/TCOMM.2021.3114682
3. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Concrete MAP Detection: A Machine Learning Inspired Relaxation,” in 24th International ITG Workshop on Smart Antennas (WSA 2020), vol. 24, Hamburg, Germany, Feb. 2020, pp. 1–5.


# Abstract of the articles

2. Following the great success of Machine Learning (ML), especially Deep Neural Networks (DNNs), in many research domains in 2010s, several ML-based approaches were proposed for detection in large inverse linear problems, e.g., massive MIMO systems. The main motivation behind is that the complexity of Maximum A-Posteriori (MAP) detection grows exponentially with system dimensions. Instead of using DNNs, essentially being a black-box, we take a slightly different approach and introduce a probabilistic Continuous relaxation of disCrete variables to MAP detection. Enabling close approximation and continuous optimization, we derive an iterative detection algorithm: Concrete MAP Detection (CMD). Furthermore, extending CMD by the idea of deep unfolding into CMDNet, we allow for (online) optimization of a small number of parameters to different working points while limiting complexity. In contrast to recent DNN-based approaches, we select the optimization criterion and output of CMDNet based on information theory and are thus able to learn approximate probabilities of the individual optimal detector. This is crucial for soft decoding in today’s communication systems. Numerical simulation results in MIMO systems reveal CMDNet to feature a promising accuracy complexity trade-off compared to State of the Art. Notably, we demonstrate CMDNet’s soft outputs to be reliable for decoders.

3. Motivated by large linear inverse problems where the complexity of the Maximum A-Posteriori (MAP) detector grows exponentially with system dimensions, e.g., large MIMO, we introduce a method to relax a discrete MAP problem into a continuous one. The relaxation is inspired by recent ML research and offers many favorable properties reflecting its quality. Hereby, we derive an iterative detection algorithm based on gradient descent optimization: Concrete MAP Detection (CMD). We show numerical results of application in large MIMO systems that demonstrate superior performance w.r.t. all considered State of the Art approaches.