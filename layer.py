class Layer:
    def __init__(self):
        self.input_nn = None
        self.output = None

    def forward(self, input_nn):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass
