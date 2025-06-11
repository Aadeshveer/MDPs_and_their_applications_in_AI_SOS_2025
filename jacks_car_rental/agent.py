import numpy as np
import matplotlib.pyplot as plt


class Agent:

    GAMMA = 0.9
    RENT = 10
    SHIFT_COST = 2
    MAX_CARS = 20
    MAX_MOVE = 5
    LOSE_PENALTY = -1000
    DELTA = 0.1

    def __init__(self, action_function):
        self.num_states = 2
        self.action_function = action_function
        self.initialize_states()

    def initialize_states(self):

        self.values = np.zeros((self.MAX_CARS+1) ** 2 + 1)
        self.pi = np.ones(((self.MAX_CARS+1)**2 + 1, 2*self.MAX_MOVE+1))
        self.pi = self.pi / self.pi.shape[1]

        self.state_to_idx = {}
        self.idx_to_state = {}

        ctr = 0

        for i in range(self.MAX_CARS + 1):

            for j in range(self.MAX_CARS + 1):

                ctr += 1
                self.state_to_idx[(i, j)] = ctr
                self.idx_to_state[ctr] = (i, j)

    def evaluate_policy(self):

        diff = np.inf

        while diff > self.DELTA:

            diff = 0

            # for all states
            for i in self.idx_to_state:
                if i % 5 == 0:
                    print(f'Evaluating policy for : {i}/{len(self.idx_to_state)}, difference is {diff}')
                v = 0

                # for all actions
                for k in range(self.pi.shape[1]):

                    probability = self.pi[i, k]

                    result = self.action_function(
                        k-self.MAX_MOVE,
                        self.idx_to_state[i],
                        self.values,
                        self.GAMMA,
                        self.state_to_idx
                    )

                    v += probability * result

                diff = max(diff, abs(v-self.values[i]))
                self.values[i] = v

    def improve_policy(self):

        # for all states
        for i in self.idx_to_state:

            if i % 20 == 0:
                print(f'Improving policy for state {i}/{len(self.idx_to_state)}')

            max_ind = [self.pi[i].argmax()]
            max_val = -np.inf

            for k in range(self.pi.shape[1]):

                result = self.action_function(
                    k-self.MAX_MOVE,
                    self.idx_to_state[i],
                    self.values,
                    self.GAMMA,
                    self.state_to_idx
                )

                new_val = result

                if (new_val > max_val):
                    max_val = new_val
                    max_ind = [k]
                elif (new_val == max_val):
                    max_ind.append(k)
            self.pi[i, ...] = 0
            for k in max_ind:
                self.pi[i, k] = 1 / len(max_ind)

    def show_values(self, save=None, slip=0):

        grid = np.zeros((self.MAX_CARS+1, self.MAX_CARS+1))
        for state in self.state_to_idx:
            grid[*state] = self.values[self.state_to_idx[state]]

        plt.figure(figsize=(8, 8))
        plt.imshow(grid, cmap='inferno')
        plt.colorbar()
        plt.xticks(np.arange(self.MAX_CARS+1))
        plt.xlabel('Cars at location 1')
        plt.yticks(np.arange(self.MAX_CARS+1))
        plt.ylabel('Cars at location 2')
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                plt.text(
                    i,
                    j,
                    '{:.2f}'.format(grid[i, j]),
                    ha='center',
                    va='center',
                    fontsize=6,
                    color='white',
                )
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
        plt.close()
