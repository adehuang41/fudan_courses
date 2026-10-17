import pickle
import numpy as np
from src.layers import Linear, Dropout
from src.activations import ReLU, Sigmoid, Tanh

class MLP:
    """Multi-layer perceptron with configurable depth, activation, and dropout."""
    def __init__(self, input_dim=784, hidden_dims=[128, 64], num_classes=10,
                 activation='relu', dropout_rate=0.0):
        self.layers = []
        self.is_training = True

        def get_activation(name):
            name = name.lower()
            if name == 'relu':    return ReLU()
            elif name == 'sigmoid': return Sigmoid()
            elif name == 'tanh':    return Tanh()
            else: raise ValueError(f"Unsupported activation: {name}")

        current_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(Linear(current_dim, h_dim))
            self.layers.append(get_activation(activation))
            if dropout_rate > 0:
                self.layers.append(Dropout(dropout_rate))
            current_dim = h_dim

        self.layers.append(Linear(current_dim, num_classes))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def train(self):
        self.is_training = True
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval(self):
        self.is_training = False
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()

    def get_params_and_grads(self):
        params_grads = []
        for layer in self.layers:
            if hasattr(layer, 'params'):
                for key in layer.params:
                    params_grads.append((layer.params[key], layer.grads[key], key))
        return params_grads

    def predict(self, x):
        was_training = self.is_training
        self.eval()
        logits = self.forward(x)
        if was_training:
            self.train()
        return np.argmax(logits, axis=1)

    def save_model(self, filepath):
        model_params = []
        for layer in self.layers:
            if hasattr(layer, 'params'):
                model_params.append({k: v.copy() for k, v in layer.params.items()})
        with open(filepath, 'wb') as f:
            pickle.dump(model_params, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_params = pickle.load(f)
        idx = 0
        for layer in self.layers:
            if hasattr(layer, 'params'):
                if idx >= len(model_params):
                    raise ValueError(
                        f"Saved file has {len(model_params)} param blocks, "
                        f"but current model expects at least {idx + 1}."
                    )
                for key in layer.params:
                    if key not in model_params[idx]:
                        raise ValueError(f"Key '{key}' missing in saved block {idx}.")
                    saved_shape = model_params[idx][key].shape
                    expected_shape = layer.params[key].shape
                    if saved_shape != expected_shape:
                        raise ValueError(
                            f"Shape mismatch for '{key}' in block {idx}: "
                            f"expected {expected_shape}, got {saved_shape}."
                        )
                    layer.params[key][...] = model_params[idx][key]
                    layer.grads[key] = np.zeros_like(layer.params[key])
                idx += 1
