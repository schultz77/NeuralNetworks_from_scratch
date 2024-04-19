import numpy as np
from layer import Layer


class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        super().__init__()

    def forward(self, input_nn):
        return np.reshape(input_nn, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)
