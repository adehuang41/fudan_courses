from abc import abstractmethod
import numpy as np


def _sliding_windows_2d(X, kernel_size, stride):
    windows = np.lib.stride_tricks.sliding_window_view(X, (kernel_size, kernel_size), axis=(2, 3))
    return windows[:, :, ::stride, ::stride, :, :]


def _linear_weight_init(in_dim, out_dim):
    scale = np.sqrt(2.0 / in_dim)
    return np.random.randn(in_dim, out_dim) * scale


def _conv_weight_init(out_channels, in_channels, kernel_size):
    fan_in = in_channels * kernel_size * kernel_size
    scale = np.sqrt(2.0 / fan_in)
    return np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
        self.trainable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=None, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if initialize_method is None:
            self.W = _linear_weight_init(in_dim, out_dim)
            self.b = np.zeros((1, out_dim))
        else:
            self.W = initialize_method(size=(in_dim, out_dim))
            self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
        self.trainable = True
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        assert self.input is not None, "Run forward before backward."
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=None, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if initialize_method is None:
            self.W = _conv_weight_init(out_channels, in_channels, kernel_size)
            self.b = np.zeros((out_channels,))
        else:
            self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
            self.b = np.zeros((out_channels,))
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}

        self.input = None
        self.padded_input = None
        self.cols = None
        self.out_h = None
        self.out_w = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.trainable = True

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out, in, k, k]
        """
        self.input = X
        if self.padding > 0:
            self.padded_input = np.pad(
                X,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            self.padded_input = X

        batch_size, _, padded_h, padded_w = self.padded_input.shape
        self.out_h = (padded_h - self.kernel_size) // self.stride + 1
        self.out_w = (padded_w - self.kernel_size) // self.stride + 1

        windows = _sliding_windows_2d(self.padded_input, self.kernel_size, self.stride)
        self.cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(batch_size * self.out_h * self.out_w, -1)
        weight_cols = self.W.reshape(self.out_channels, -1)
        output = self.cols @ weight_cols.T + self.b
        return output.reshape(batch_size, self.out_h, self.out_w, self.out_channels).transpose(0, 3, 1, 2)

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        assert self.padded_input is not None, "Run forward before backward."

        grads_2d = grads.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        weight_cols = self.W.reshape(self.out_channels, -1)

        grad_W = grads_2d.T @ self.cols
        grad_input_cols = grads_2d @ weight_cols
        batch_size = self.padded_input.shape[0]
        grad_input_cols = grad_input_cols.reshape(batch_size, self.out_h, self.out_w, self.in_channels, self.kernel_size, self.kernel_size)
        grad_input_cols = grad_input_cols.transpose(0, 3, 1, 2, 4, 5)

        grad_input_padded = np.zeros_like(self.padded_input)
        for kh in range(self.kernel_size):
            h_slice = slice(kh, kh + self.out_h * self.stride, self.stride)
            for kw in range(self.kernel_size):
                w_slice = slice(kw, kw + self.out_w * self.stride, self.stride)
                grad_input_padded[:, :, h_slice, w_slice] += grad_input_cols[:, :, :, :, kh, kw]

        self.grads['W'] = grad_W.reshape(self.W.shape)
        self.grads['b'] = np.sum(grads_2d, axis=0)

        if self.padding > 0:
            return grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return grad_input_padded
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
        self.argmax = None
        self.out_h = None
        self.out_w = None
        self.optimizable = False
        self.trainable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        windows = _sliding_windows_2d(X, self.kernel_size, self.stride)
        self.out_h = windows.shape[2]
        self.out_w = windows.shape[3]
        flat_windows = windows.reshape(*windows.shape[:4], -1)
        self.argmax = np.argmax(flat_windows, axis=-1)
        output = np.take_along_axis(flat_windows, self.argmax[..., None], axis=-1)[..., 0]
        return output

    def backward(self, grads):
        assert self.input is not None and self.argmax is not None, "Run forward before backward."
        grad_input = np.zeros_like(self.input)
        grad_windows = np.zeros((grads.shape[0], grads.shape[1], self.out_h, self.out_w, self.kernel_size * self.kernel_size), dtype=grads.dtype)
        np.put_along_axis(grad_windows, self.argmax[..., None], grads[..., None], axis=-1)
        grad_windows = grad_windows.reshape(grads.shape[0], grads.shape[1], self.out_h, self.out_w, self.kernel_size, self.kernel_size)

        for kh in range(self.kernel_size):
            h_slice = slice(kh, kh + self.out_h * self.stride, self.stride)
            for kw in range(self.kernel_size):
                w_slice = slice(kw, kw + self.out_w * self.stride, self.stride)
                grad_input[:, :, h_slice, w_slice] += grad_windows[:, :, :, :, kh, kw]
        return grad_input
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False
        self.trainable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.probs = None
        self.grads = None
        self.optimizable = False
        self.trainable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels

        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts

        batch_size = predicts.shape[0]
        clipped_probs = np.clip(self.probs, 1e-12, 1.0)
        return -np.mean(np.log(clipped_probs[np.arange(batch_size), labels]))

    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.predicts.shape[0]
        grads = self.probs.copy()
        grads[np.arange(batch_size), self.labels] -= 1.0
        grads /= batch_size
        self.grads = grads
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
