import numpy as np
import matplotlib.pyplot as plt


class Gambler:

    DELTA = 0.001
    GAMMA = 1

    def __init__(self, prob_head=0.4, num_states=100):
        self.ph = prob_head
        self.num_states = num_states
        self.values = np.zeros(num_states-1)
        self.pi = np.ones((num_states-1, num_states//2 + 1)) / (num_states//2)
        self.pi[:num_states//2] = np.tri(num_states//2, num_states//2 + 1)
        for i in range(num_states//2):
            self.pi[i] /= (i+1)

    def iterate_values(self):

        diff = np.inf
        ctr = 0

        plt.figure(figsize=(8, 8))
        while (diff > self.DELTA):

            diff = 0
            val = self.values
            # for all states
            for i in range(val.shape[0]):
                # state i has i+1 coins
                print(f'Processing {i}th state')

                v = 0

                # for all actions
                max_idx = []
                max_val = -1
                for j in range(self.pi.shape[1]):
                    # bet j coins
                    if j <= i:
                        temp = 0
                        if i+j+1 >= self.num_states:
                            temp += self.ph * 1
                        else:
                            temp += self.ph * val[j+i] * self.GAMMA
                        if i+1-j <= 0:
                            temp += 0
                        else:
                            temp += (1-self.ph) * val[i-j] * self.GAMMA
                        if temp > max_val:
                            max_idx = [j]
                            max_val = temp
                        elif temp == max_val:
                            max_idx.append(j)
                v = max_val
                self.pi[i] = 0
                for bet in max_idx:
                    self.pi[i, bet] = 1/len(max_idx)

                diff = max(diff, abs(v-val[i]))
                self.values[i] = v
            plt.title(f'Values with probability {self.ph}')
            ctr += 1
            plt.plot(self.values, label=f'Sweep:{ctr}')
        plt.xlabel('Capital')
        plt.ylabel('Value Estimates')
        plt.legend()
        plt.savefig(f'results/values_prob_{int(self.ph*100)}_percent.png')
        # policy
        plt.figure(figsize=(8, 8))
        plt.title(f'Policy with probability {self.ph}')
        plt.xlabel('Capital')
        plt.ylabel('Final policy')
        grid = np.zeros(self.pi.shape[0])
        for i in range(grid.shape[0]):
            grid[i] = np.average(
                np.arange(self.pi.shape[1]), weights=self.pi[i]
            )
        plt.plot(grid)
        plt.savefig(f'results/policy_prob_{int(100*prob)}_percent.png')


for prob in [0.25, 0.4, 0.55]:
    gambler = Gambler(prob)
    gambler.iterate_values()
