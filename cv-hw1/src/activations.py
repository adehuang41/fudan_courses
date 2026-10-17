import numpy as np

class Activation:
    def __init__(self):
        pass

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError


class ReLU(Activation):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return np.maximum(0, x)

    def backward(self, dout):
        return dout * self.mask


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self.out

    def backward(self, dout):
        # f'(x) = f(x) * (1 - f(x))
        return dout * self.out * (1.0 - self.out)


class Tanh(Activation):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        # f'(x) = 1 - f(x)^2
        return dout * (1.0 - self.out ** 2)


class Softmax(Activation):
    """Softmax for inference only; backward is fused into CrossEntropyLoss."""
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out

    def backward(self, dout):
        raise RuntimeError("Softmax backward should be fused with CrossEntropyLoss.")
