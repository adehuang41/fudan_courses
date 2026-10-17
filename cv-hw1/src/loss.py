import numpy as np

class Loss:
    def __init__(self):
        pass

    def forward(self, logits, y_true):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    """Cross-entropy loss with fused softmax for numerical stability."""
    def __init__(self):
        super().__init__()
        self.probs = None
        self.y_true = None

    def forward(self, logits, y_true):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted)
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.y_true = y_true
        batch_size = logits.shape[0]
        safe_probs = np.clip(self.probs, 1e-15, 1.0 - 1e-15)
        return -np.sum(y_true * np.log(safe_probs)) / batch_size

    def backward(self):
        return (self.probs - self.y_true) / self.probs.shape[0]
