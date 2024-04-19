import numpy as np
from layer import Layer


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        """
        :param activation: activation function
        :param activation_prime: activation-function derivative
        """
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_nn):
        self.input_nn = input_nn
        return self.activation(self.input_nn)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input_nn))
