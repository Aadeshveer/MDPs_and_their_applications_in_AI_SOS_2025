import numpy as np
import scipy  # for poisson probability
from agent import Agent


class Env:

    RENT = 10
    SHIFT_COST = 2
    MAX_CARS = 20
    MAX_MOVE = 5
    MAX_ITERATION = 20
    PARKING_COST = 4
    PARKING_LIMIT = 10

    def __init__(self, updated_test=False):
        self.out_lambda = np.array([3, 4])
        self.in_lambda = np.array([2, 3])
        self.next_returns = np.zeros(2)
        self.agent = Agent(self.simulate_day)
        self.updated_test = updated_test

    def simulate_day(self, movement, cars, values, gamma):

        result = 0
        shape = (
            self.MAX_ITERATION,
            self.MAX_ITERATION,
            self.MAX_ITERATION,
            self.MAX_ITERATION,
        )
        ncars = np.zeros((
            2,
            *shape
        ))
        ncars[0] = cars[0]
        ncars[1] = cars[1]
        credits = np.zeros(shape)

        # all possible next states
        prob1 = scipy.stats.poisson.pmf(
            np.arange(self.MAX_ITERATION), self.out_lambda[0]
        ).reshape((-1, 1, 1, 1))
        prob2 = scipy.stats.poisson.pmf(
            np.arange(self.MAX_ITERATION), self.out_lambda[1]
        ).reshape((1, -1, 1, 1))
        prob3 = scipy.stats.poisson.pmf(
            np.arange(self.MAX_ITERATION), self.in_lambda[0]
        ).reshape((1, 1, -1, 1))
        prob4 = scipy.stats.poisson.pmf(
            np.arange(self.MAX_ITERATION), self.in_lambda[1]
        ).reshape((1, 1, 1, -1))

        ncars[0] -= movement
        ncars[1] += movement
        credits -= self.SHIFT_COST * abs(movement)

        # including the special worker case
        if self.updated_test and movement >= 1:
            credits += self.SHIFT_COST

        ncars[0] = np.clip(ncars[0], 0, self.MAX_CARS)
        ncars[1] = np.clip(ncars[1], 0, self.MAX_CARS)

        rented_1 = np.minimum(
            np.arange(self.MAX_ITERATION).reshape(-1, 1, 1, 1),
            ncars[0]
        )
        rented_2 = np.minimum(
            np.arange(self.MAX_ITERATION).reshape(1, -1, 1, 1),
            ncars[1]
        )
        return_1 = np.arange(self.MAX_ITERATION).reshape(1, 1, -1, 1)
        return_2 = np.arange(self.MAX_ITERATION).reshape(1, 1, 1, -1)

        ncars[0] = ncars[0] + return_1 - rented_1
        ncars[1] = ncars[1] + return_2 - rented_2

        # adding parking case
        if self.updated_test:
            credits -= (ncars[0] > self.PARKING_LIMIT)*self.PARKING_COST
            credits -= (ncars[1] > self.PARKING_LIMIT)*self.PARKING_COST

        credits += (self.RENT*(rented_1+rented_2))

        ncars[0] = np.clip(ncars[0], 0, self.MAX_CARS)
        ncars[1] = np.clip(ncars[1], 0, self.MAX_CARS)

        ncars = ncars.astype(np.int32)

        future_value = values[ncars[0], ncars[1]]

        result = prob1*prob2*prob3*prob4 * (credits + gamma * future_value)

        return result.sum()


# world = Env()
# world.agent.iterate_policy(True)
# world.agent.show_values('results/value_function.png')

# world = Env()
# world.agent.iteratate_values()
# world.agent.show_values('results/value_iteration.png')

world = Env(True)
world.agent.iteratate_values()
world.agent.improve_policy()
world.agent.show_policy('results/exercise_policy.png')
world.agent.show_values('results/exercise_values.png')
