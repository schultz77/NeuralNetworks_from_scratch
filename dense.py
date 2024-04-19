from layer import Layer
import numpy as np


class Dense(Layer):
    def __init__(self, input_size, output_size):
        """

        :param input_size: number of neurons at the input
        :param output_size: number of neuron at the output
        """
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input_nn):
        self.input_nn = input_nn

        return np.dot(self.weights, self.input_nn) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input_nn.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        # gradient descent
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient

        return input_gradient
