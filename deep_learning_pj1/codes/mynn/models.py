from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplemented
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self):
        self.layers = [
            conv2D(1, 8, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            conv2D(8, 16, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Linear(16 * 7 * 7, 128),
            ReLU(),
            Linear(128, 10),
        ]
        self._named_layers = {
            'conv1': self.layers[0],
            'conv2': self.layers[3],
            'fc1': self.layers[6],
            'fc2': self.layers[8],
        }

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = X
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, Linear) and outputs.ndim == 4:
                outputs = outputs.reshape(outputs.shape[0], -1)
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
            if isinstance(layer, Linear) and grads.ndim == 2 and layer is self.layers[6]:
                grads = grads.reshape(grads.shape[0], 16, 7, 7)
        return grads

    def set_trainable_policy(self, policy):
        policy = policy.lower()
        trainable_map = {
            'all': {'conv1', 'conv2', 'fc1', 'fc2'},
            'head_only': {'fc1', 'fc2'},
            'conv2_head': {'conv2', 'fc1', 'fc2'},
        }
        if policy not in trainable_map:
            raise ValueError(f'Unknown trainable policy: {policy}')
        active = trainable_map[policy]
        for name, layer in self._named_layers.items():
            layer.trainable = name in active

    def get_trainable_layer_names(self):
        return [name for name, layer in self._named_layers.items() if layer.trainable]

    def get_frozen_layer_names(self):
        return [name for name, layer in self._named_layers.items() if not layer.trainable]

    def count_parameters(self, trainable_only=False):
        total = 0
        for layer in self.layers:
            if not layer.optimizable:
                continue
            if trainable_only and not getattr(layer, 'trainable', True):
                continue
            total += int(layer.params['W'].size + layer.params['b'].size)
        return total

    def trainable_parameter_summary(self):
        total = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)
        return {
            'trainable_layers': '+'.join(self.get_trainable_layer_names()),
            'frozen_layers': '+'.join(self.get_frozen_layer_names()),
            'trainable_param_count': trainable,
            'total_param_count': total,
            'trainable_param_ratio': (trainable / total) if total > 0 else 0.0,
        }
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)

        self.__init__()
        optimizable_layers = [layer for layer in self.layers if layer.optimizable]
        for layer, saved in zip(optimizable_layers, param_list):
            layer.W = saved['W']
            layer.b = saved['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = saved['weight_decay']
            layer.weight_decay_lambda = saved['lambda']
        
    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
