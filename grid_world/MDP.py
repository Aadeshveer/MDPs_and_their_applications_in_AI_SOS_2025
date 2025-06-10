import enum
import random
import numpy as np
import matplotlib.pyplot as plt


class Action(enum.Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


class GridWorldMDP:

    DELTA = 0.001

    def __init__(self, mapping, normal_value, action_function, gamma):
        self.map = mapping
        self.SIZE = self.map.shape
        self.action_function = action_function
        self.init_value_function(normal_value)
        self.gamma = gamma

    def init_value_function(self, normal_value):
        self.state_loc_map = {}
        self.loc_state_map = {}
        ctr = 0
        for i in range(self.SIZE[0]):
            for j in range(self.SIZE[1]):
                if self.map[i][j] == normal_value:
                    self.state_loc_map[ctr] = (i, j)
                    self.loc_state_map[(i, j)] = ctr
                    ctr += 1
        self.values = np.zeros(ctr)
        self.pi = np.ones((ctr, 4)) / 4

    def evaluate_policy(self):
        diff = np.inf
        while diff > self.DELTA:
            diff = 0
            for i in range(self.values.size):
                v = 0
                for action in [
                    Action.LEFT,
                    Action.UP,
                    Action.RIGHT,
                    Action.DOWN
                ]:
                    probability = self.pi[i][action.value]
                    new_pos, reward = self.action_function(
                        action,
                        self.state_loc_map[i]
                    )
                    new_disc_reward = self.gamma * self.values[
                        self.loc_state_map[new_pos]
                    ]
                    v += probability * (reward + new_disc_reward)
                diff = max(diff, abs(v - self.values[i]))
                self.values[i] = v

    def improve_policy(self):
        for i in range(self.values.size):
            max_ind = [self.pi[i].max()]
            max_val = -np.inf
            action_list = [
                Action.LEFT,
                Action.UP,
                Action.RIGHT,
                Action.DOWN
            ]
            random.shuffle(action_list)
            for action in action_list:
                new_pos, reward = self.action_function(
                    action,
                    self.state_loc_map[i]
                )
                new_disc_reward = self.gamma * self.values[
                    self.loc_state_map[new_pos]
                ]
                new_val = reward + new_disc_reward
                if (new_val > max_val):
                    max_val = new_val
                    max_ind = [action.value]
                elif (new_val == max_val):
                    max_ind.append(action.value)
            self.pi[i, ...] = 0
            for j in max_ind:
                self.pi[i, j] = 1 / len(max_ind)

    def iterate_policy(self):
        old_policy = self.pi.copy()
        self.evaluate_policy()
        self.improve_policy()
        while (old_policy != self.pi).any():
            old_policy = self.pi.copy()
            self.evaluate_policy()
            self.improve_policy()

    def iterate_values(self):
        diff = np.inf
        while diff > self.DELTA:
            diff = 0
            new_values = np.zeros(4)
            for i in range(self.values.size):
                v = 0
                for action in [
                    Action.LEFT,
                    Action.UP,
                    Action.RIGHT,
                    Action.DOWN
                ]:
                    new_pos, reward = self.action_function(
                        action,
                        self.state_loc_map[i]
                    )
                    new_disc_reward = self.gamma * self.values[
                        self.loc_state_map[new_pos]
                    ]
                    new_values[action.value] = reward + new_disc_reward
                v = new_values.max()
                diff = max(diff, abs(v - self.values[i]))
                self.values[i] = v

    def show_values(self, save=None, slip=0):
        grid = self.map
        for state in self.state_loc_map:
            grid[*self.state_loc_map[state]] = self.values[state]
        plt.imshow(grid.T, cmap='inferno')
        plt.colorbar()
        plt.title(f'''Expected values for various states
gamma:{self.gamma}, slip probability:{slip}''')
        plt.xticks(np.arange(self.SIZE[0]))
        plt.yticks(np.arange(self.SIZE[1]))
        for i in range(self.SIZE[0]):
            for j in range(self.SIZE[1]):
                if self.map[i, j] != -1:
                    plt.text(
                        i,
                        j,
                        '{:.2f}'.format(grid[i, j]),
                        ha='center',
                        va='center',
                        fontsize=8,
                        color='white',
                    )
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
        plt.close()

    def show_policy(self, save=None, slip=0):
        plt.imshow(self.map.T)
        plt.title(f'''Policy solving GridWorld problem
gamma:{self.gamma}, slip probability:{slip}''')
        plt.xticks(np.arange(self.SIZE[0]))
        plt.yticks(np.arange(self.SIZE[1]))
        for (i, j) in self.loc_state_map:
            dx = 0
            dy = 0
            prob_dist = self.pi[self.loc_state_map[(i, j)]].copy()
            max = prob_dist.max()
            while (max == prob_dist.max()):
                match np.argmax(prob_dist):
                    case 0:
                        dx = -1
                        prob_dist[0] = -1
                    case 1:
                        dy = -1
                        prob_dist[1] = -1
                    case 2:
                        dx = 1
                        prob_dist[2] = -1
                    case 3:
                        dy = 1
                        prob_dist[3] = -1
                plt.arrow(i, j, dx*0.2, dy*0.2, color='white', width=0.05)
                dx = 0
                dy = 0
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
        plt.close()
