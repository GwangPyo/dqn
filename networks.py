import torch
import torch.nn as nn
import numpy as np
import gym.spaces


class MlpNet(nn.Module):
    def __init__(self, input_size, output_size, activation=nn.ReLU, **kwargs):
        super().__init__()
        self.net = nn.Linear(input_size, output_size, **kwargs)
        self.activation = activation()

    def forward(self, x):
        x = self.net(x)
        return self.activation(x)


class MlpDQNLayer(nn.Sequential):
    def __init__(self, obs_space, action_space, layer_kwargs=None, layer_size=None,):
        self.input_size = np.product(obs_space.shape)
        assert isinstance(action_space, gym.spaces.Discrete)
        self.output_size = action_space.n
        if layer_kwargs is None:
            layer_kwargs = {}
        if layer_size is None:
            layer_size = [self.input_size, 64, 64]
        else:
            layer_size = [self.input_size] + layer_size

        layers = []
        for i in range(len(layer_size) - 1):
            input_size = layer_size[i]
            output_size = layer_size[i + 1]
            layers.append(MlpNet(input_size, output_size, **layer_kwargs))
        layers.append(nn.Linear(layer_size[-1], self.output_size, **layer_kwargs))
        super(MlpDQNLayer, self).__init__(*layers)



