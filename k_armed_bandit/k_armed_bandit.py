import numpy as np


class KArmedBandit:

    def __init__(self, k=10, std_dev=1, means=None):
        self.k = k
        self.std_dev = std_dev
        if means is None:
            self.means = 2 * np.random.random(size=k) - 1
        else:
            self.means = means

    def draw(self, i):
        return np.random.normal(loc=self.means[i], scale=self.std_dev)

    def get_optimal(self):
        return np.argmax(self.means)


class EpsilonGreedy:

    def __init__(self, epsilon=0.1, k=10):
        self.epsilon = epsilon
        self.k = k
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def choose_action(self):
        if np.random.random() < self.epsilon:
            prediction = np.random.randint(0, self.k)
        else:
            prediction = np.argmax(self.Q)
        return prediction

    def update_table(self, i, reward):
        self.N[i] += 1
        self.Q[i] = self.Q[i] + (reward - self.Q[i]) / self.N[i]
