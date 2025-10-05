import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class Plotter:

    def __init__(self) -> None:
        self.score_list: list[int] = []
        self.average_list: list[float] = []
        self.num_iter: list[int] = []
        self.average_iter: list[float] = []
        self.iter_ctr: NDArray[np.int16] = np.zeros(1000, dtype=np.int16)
        self.limit = 100

    def add_score(self, score: int, ctr: int):
        self.score_list.append(score)
        self.num_iter.append(ctr)
        self.iter_ctr[ctr] += 1
        while self.limit < ctr:
            self.limit += 100
        if len(self.average_list) == 0:
            average: float = score
        else:
            average: float = self.average_list[-1]
            average += (score-self.average_list[-1])/(len(self.score_list))
        self.average_list.append(average)
        if len(self.average_iter) == 0:
            average: float = ctr
        else:
            average: float = self.average_iter[-1]
            average += (ctr-self.average_iter[-1])/(len(self.num_iter))
        self.average_iter.append(average)

    def plot(self, file_name: str | None):
        plt.figure(figsize=(12, 8), dpi=600)

        plt.subplot(2, 1, 1)
        plt.title('performance')
        plt.xlabel('time')
        plt.ylabel('score')
        plt.plot(self.score_list, label='Scores')
        plt.plot(self.average_list, label='Average')
        plt.legend()

        plt.subplot(2, 3, 4)
        plt.xlabel('score')
        plt.ylabel('number')
        plt.title('score distribution')
        plt.hist(self.score_list)

        plt.subplot(2, 3, 5)
        plt.ylabel('average num of iterations')
        plt.xlabel('time')
        plt.title('iteration counter')
        plt.plot(self.average_iter)

        plt.subplot(2, 3, 6)
        plt.ylabel('num of times arrived on')
        plt.xlabel('episode length')
        plt.title('Episode length')
        plt.plot(np.arange(self.limit), self.iter_ctr[:self.limit], label='num of iterations')
        plt.set_yscale('log')

        plt.tight_layout()
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name)
        plt.close()