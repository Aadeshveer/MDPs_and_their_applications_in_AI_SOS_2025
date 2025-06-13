import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # for generating beautiful heatmaps


class Agent:

    GAMMA = 0.9
    MAX_CARS = 20
    MAX_MOVE = 5
    DELTA = 0.01

    def __init__(self, action_function):
        self.action_function = action_function
        self.initialize_states()

    def initialize_states(self):

        self.values = np.zeros(
            (
                self.MAX_CARS+1,
                self.MAX_CARS+1
            )
        )
        self.pi = np.ones(
            (
                self.MAX_CARS+1,
                self.MAX_CARS+1,
                2*self.MAX_MOVE+1
            )
        )
        self.pi = self.pi / self.pi.shape[2]

    def evaluate_policy(self, synchronous=False):

        diff = np.inf

        while diff > self.DELTA:

            diff = 0

            temp_values = self.values.copy() if synchronous else self.values

            # for all states
            for i in range(temp_values.shape[0]):
                print(f'Evaluating policy for : {i}/{temp_values.shape[0]}, difference is {diff}')  # Noqa:E501
                for j in range(temp_values.shape[1]):
                    v = 0

                    # for all actions
                    for k in range(self.pi.shape[2]):

                        probability = self.pi[i, j, k]

                        result = self.action_function(
                            k-self.MAX_MOVE,
                            (i, j),
                            temp_values,
                            self.GAMMA,
                        )
                        v += probability * result
                    diff = max(diff, abs(v-temp_values[i, j]))
                    self.values[i, j] = v

    def improve_policy(self):

        # for all states
        for i in range(self.values.shape[0]):
            print(f'Improving policy for : {i}/{self.values.shape[0]}')
            for j in range(self.values.shape[1]):

                max_ind = [self.pi[i, j].argmax()]
                max_val = -np.inf

                for k in range(self.pi.shape[2]):
                    if k-self.MAX_MOVE > i or -k+self.MAX_MOVE > j:
                        continue
                    result = self.action_function(
                        k-self.MAX_MOVE,
                        (i, j),
                        self.values,
                        self.GAMMA,
                    )

                    new_val = result

                    if (new_val > max_val):
                        max_val = new_val
                        max_ind = [k]
                    elif (new_val == max_val):
                        max_ind.append(k)
                self.pi[i, j, ...] = 0
                for k in max_ind:
                    self.pi[i, j, k] = 1 / len(max_ind)

    def iterate_policy(self, synchronous=False, save=None):
        old_policy = self.pi.copy()
        ctr = 1
        self.evaluate_policy(synchronous)
        self.improve_policy()
        if synchronous:
            self.show_policy(f'results/policy_after_iteration{ctr}.png')
        while (old_policy != self.pi).any():
            ctr += 1
            old_policy = self.pi.copy()
            self.evaluate_policy(synchronous)
            self.improve_policy()
            if synchronous:
                self.show_policy(f'results/policy_after_iteration{ctr}.png')

    def iteratate_values(self):

        diff = np.inf

        while diff > self.DELTA:

            diff = 0

            temp_values = self.values

            # for all states
            for i in range(temp_values.shape[0]):
                print(f'Evaluating policy for : {i}/{temp_values.shape[0]}, difference is {diff}')  # Noqa:E501
                for j in range(temp_values.shape[1]):
                    v = 0

                    # for all actions
                    for k in range(self.pi.shape[2]):

                        result = self.action_function(
                            k-self.MAX_MOVE,
                            (i, j),
                            temp_values,
                            self.GAMMA,
                        )
                        v = max(result, v)
                    diff = max(diff, abs(v-temp_values[i, j]))
                    self.values[i, j] = v

    def show_values(self, save=None):

        plt.figure(figsize=(8, 8))
        plot_object = sns.heatmap(
            np.rot90(self.values),
            square=True,
            yticklabels=np.arange(self.MAX_CARS, -1, -1),
            xticklabels=np.arange(self.MAX_CARS + 1),
        )
        plot_object.set(
            xlabel='Cars at location 2',
            ylabel='Cars at location 1',
            title='Values'
        )
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
        plt.close()

    def show_policy(self, save=None):
        grid = np.zeros((self.MAX_CARS+1, self.MAX_CARS+1))
        for i in range(self.MAX_CARS+1):
            for j in range(self.MAX_CARS+1):
                grid[i, j] = np.average(
                    np.arange(-self.MAX_MOVE, self.MAX_MOVE+1),
                    weights=self.pi[i, j]
                )
        grid = np.rot90(grid.T)
        plt.figure(figsize=(8, 8))
        plot_object = sns.heatmap(
            grid,
            square=True,
            yticklabels=np.arange(self.MAX_CARS, -1, -1),
            xticklabels=np.arange(self.MAX_CARS + 1),
        )
        plot_object.set(
            xlabel='Cars at location 2',
            ylabel='Cars at location 1',
            title='Policy'
        )
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
        plt.close()
