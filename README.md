[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liamnaka/space/)

Extends [HoloGAN](https://www.monkeyoverflow.com/hologan-unsupervised-learning-of-3d-representations-from-natural-images/) by adding a subnetwork to the generator that outputs a pose distribution conditioned on *z*. This generated distribution is sampled to pose the corresponding 3D representation before yielding an output image.

Like HoloGAN, the model is currently limited to *SO(3)* transformation of the 3D representation.
