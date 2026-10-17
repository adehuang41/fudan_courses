import numpy as np

class Layer:
    """Base class for network layers."""
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.is_training = True

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False


class Linear(Layer):
    """Fully connected layer."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        scale = np.sqrt(2.0 / input_dim)  # He initialization
        self.params['W'] = np.random.randn(input_dim, output_dim) * scale
        self.params['b'] = np.zeros((1, output_dim))
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.params['W']) + self.params['b']

    def backward(self, dout):
        self.grads['W'] = np.dot(self.inputs.T, dout)
        self.grads['b'] = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.params['W'].T)


class Dropout(Layer):
    """Inverted dropout layer."""
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, inputs):
        if self.is_training:
            self.mask = (np.random.rand(*inputs.shape) > self.dropout_rate)
            return inputs * self.mask / (1.0 - self.dropout_rate)
        return inputs

    def backward(self, dout):
        if not self.is_training:
            return dout
        return dout * self.mask / (1.0 - self.dropout_rate)
