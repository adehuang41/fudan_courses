import numpy as np

class Optimizer:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """SGD with momentum and optional L2 weight decay."""
    def __init__(self, model, lr=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(model, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = None

    def step(self):
        params_grads = self.model.get_params_and_grads()

        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p, _, _ in params_grads]

        for i, (param, grad, key) in enumerate(params_grads):
            # L2 weight decay applies only to weight matrices, not biases
            if self.weight_decay > 0 and key == 'W':
                g = grad + self.weight_decay * param
            else:
                g = grad

            # v = momentum * v + g
            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] + g
                update = self.velocities[i]
            else:
                update = g

            param -= self.lr * update


class ExponentialScheduler:
    """Exponential learning rate decay, optionally staircase."""
    def __init__(self, optimizer, decay_rate=0.95, decay_steps=500, staircase=True):
        self.optimizer = optimizer
        self.initial_lr = optimizer.lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.staircase = staircase
        self.current_step = 0

    def step(self):
        self.current_step += 1
        stage = self.current_step / self.decay_steps
        if self.staircase:
            stage = np.floor(stage)
        self.optimizer.lr = self.initial_lr * (self.decay_rate ** stage)
