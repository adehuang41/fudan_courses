from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True and getattr(layer, 'trainable', True):
                for key in layer.params.keys():
                    if key == 'W' and layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]
                layer.W = layer.params['W']
                layer.b = layer.params['b']


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocity = {}
        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                self.velocity[layer_idx] = {
                    key: np.zeros_like(value) for key, value in layer.params.items()
                }
    
    def step(self):
        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable and getattr(layer, 'trainable', True):
                for key in layer.params.keys():
                    if key == 'W' and layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    self.velocity[layer_idx][key] = self.mu * self.velocity[layer_idx][key] - self.init_lr * layer.grads[key]
                    layer.params[key] = layer.params[key] + self.velocity[layer_idx][key]
                layer.W = layer.params['W']
                layer.b = layer.params['b']
