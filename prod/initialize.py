import os
import numpy as np
from keras import backend as K
from keras.initializers import Initializer

class LightLoadInitializer(Initializer):
    def __init__(self, layer_name, bias, weights_dir):
        self.layer_name = layer_name
        self.bias = bias
        self.weights_dir= weights_dir

    def __call__(self, shape=None, dtype=None):
        # Return a Tensor, with the same values as the weights / biases
        if self.bias:
            biases = np.load(os.path.join(self.weights_dir, '%s_biases.npy' % self.layer_name))
            return K.variable(value=biases)
        else:
            weights = np.load(os.path.join(self.weights_dir, '%s_weights.npy' % self.layer_name))
            return K.variable(value=weights)

    def get_config(self):
        return {
            'layer_name': self.layer_name,
            'bias': self.bias,
            'weights_dir': self.weights_dir,
        }
