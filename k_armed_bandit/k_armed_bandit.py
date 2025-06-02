import numpy as np


class KArmedBandit:

    def __init__(self, k=10, stationary=True, std_dev=1, means=None):
        self.k = k
        self.std_dev = std_dev
        self.stationary = stationary
        if means is None:
            if self.stationary:
                self.means = 2 * np.random.random(size=k) - 1
            else:
                self.means = np.zeros(k)
        else:
            self.means = means

    def draw(self, i):
        if not self.stationary:
            self.means += np.random.normal(loc=0, scale=0.01, size=self.k)
        return np.random.normal(loc=self.means[i], scale=self.std_dev)

    def get_optimal(self):
        return np.argmax(self.means)


class EpsilonGreedy:

    def __init__(
        self,
        epsilon=0.1,
        k=10,
        stationary=True,
        alpha=0.1,
        Q_initialization=0,
        baseline=0
    ):
        self.epsilon = epsilon
        self.k = k
        self.stationary = stationary
        self.alpha = alpha
        self.Q = np.ones(k) * Q_initialization
        self.N = np.zeros(k)
        self.H = np.zeros(k)
        self.pi = np.ones(k) / k
        self.R_mean = baseline

    def choose_action(self, UCB=False, Gradient=False, c=2):
        if UCB:
            return np.argmax(
                self.Q + c * np.sqrt(np.log(self.N.sum()) / self.N)
            )
        if Gradient:
            return np.random.choice(np.arange(self.k), p=self.pi)
        if np.random.random() < self.epsilon:
            prediction = np.random.randint(0, self.k)
        else:
            prediction = np.argmax(self.Q)
        return prediction

    def update_table(self, i, reward, grad=False):
        self.N[i] += 1
        if grad:
            self.H = self.H - self.alpha * (reward - self.R_mean) * (self.pi)
            self.H[i] += self.alpha * (reward - self.R_mean)
            self.pi = np.exp(self.H) / np.sum(np.exp(self.H))
            self.R_mean = self.R_mean + (reward - self.R_mean) / self.N.sum()
        else:
            alpha_factor = (1 / self.N[i] if self.stationary else self.alpha)
            self.Q[i] = self.Q[i] + (reward - self.Q[i]) * alpha_factor
