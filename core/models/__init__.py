from .autoencoder import AutoEncoder
from .baseline_MNIST_network import BaselineMNISTNetwork
from .resnet import ResNet
from .vgg import *
from .resnetspec import ResNetspec
from .resnetspecdct import ResNetdct
from .resnetspecdct2 import ResNetdct2
from .resnet224 import ResNet224
__all__ = [
    'AutoEncoder', 'BaselineMNISTNetwork', 'ResNet','ResNetspec','ResNetdct','ResNetdct2','ResNet224',
]