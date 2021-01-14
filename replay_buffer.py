import numpy as np


class ReplayBuffer(object):
    def __init__(self, maxlen=50000):
        self.maxlen = maxlen
        self._len = 0
        self.index = -1
        self._storage = np.full(shape=(self.maxlen, 5), fill_value=[None, None, None, None, None], dtype=object)

    def add(self, state, action, reward, done, next_state):
        self.index = (self.index + 1) % self.maxlen
        transition = [np.copy(state), np.copy(action), reward, int(done), np.copy(next_state)]
        self._storage[self.index] = transition
        self._len = min(self._len + 1, self.maxlen)

    def get(self, batch_size):
        random_index = np.random.randint(low=0, high=self._len, size=(batch_size, ))
        transitions = self._storage[random_index]
        states = np.vstack(transitions[:, 0])
        actions = np.vstack(transitions[:, 1])
        rewards = np.vstack(transitions[:, 2])
        dones = np.vstack(transitions[:, 3])
        next_state = np.vstack(transitions[:, 4])
        return np.asarray(states,  dtype=np.float32), np.asarray(actions, dtype=np.int32), \
               np.asarray(dones, dtype=np.int32),\
               np.asarray(rewards, dtype=np.float32), np.asarray(next_state, dtype=np.float32)

    def __len__(self):
        return self._len


