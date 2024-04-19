import numpy as np
from scipy import signal
from layer import Layer


class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        """

        :param input_shape: tuple containing the depth, height and width of the input
        :param kernel_size: number representing the size of each matrix inside each kernel
        :param depth: number representing how many kernels are set
        """
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

        super().__init__()

    def forward(self, input_nn):
        self.input_nn = input_nn
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input_nn[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input_nn[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient  # gradient descent
        self.biases -= learning_rate * output_gradient  # gradient descent
        return input_gradient
