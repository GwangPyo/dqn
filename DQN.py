import numpy as np
import networks
import replay_buffer
import torch
import gym


class DQN(object):
    """

    """

    def __init__(self, env, gamma=0.99):

        self.env = env
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space.n
        self.policy = networks.MlpDQNLayer(env.observation_space, env.action_space)
        self.double = networks.MlpDQNLayer(env.observation_space, env.action_space)
        self.replay_buffer = replay_buffer.ReplayBuffer()
        self.gamma = gamma

    def learn_step(self, batch_size):
        if batch_size < len(self.replay_buffer):
            return
        else:
            obs, action, reward, dones, next_obs = self.replay_buffer.get(batch_size)
            y_tp1 = self.double.predict(obs).detach()
            dones = self.gamma * torch.from_numpy(dones)
            """
            if not done:
               y_hat = gamma * Q(s_{t + 1}, a) if s_{t + 1} 
            """
            #
            y_tp1 = torch.multiply(y_tp1, dones)

