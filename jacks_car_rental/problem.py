import numpy as np
from agent import Agent


class Env:

    RENT = 10
    SHIFT_COST = 2
    MAX_CARS = 20
    MAX_MOVE = 5
    LOSE_PENALTY = -10
    SAMPLE_SIZE = 100
    MAX_ITERATION = 10

    def __init__(self):
        self.out_lambda = np.array([3, 4])
        self.in_lambda = np.array([2, 3])
        self.next_returns = np.zeros(2)
        self.agent = Agent(self.simulate_day)

    def simulate_day(self, movement, cars, values, gamma, state_to_index):

        prob = 1
        result = 0
        ncars = list(cars)

        # all possible next states
        for request_1 in range(self.MAX_ITERATION):
            prob1 = prob * self.poisson_probability(
                request_1, self.out_lambda[0]
            )

            for request_2 in range(self.MAX_ITERATION):
                prob2 = prob1 * self.poisson_probability(
                    request_2,
                    self.out_lambda[1]
                )

                for returns_1 in range(self.MAX_ITERATION):
                    prob3 = prob2 * self.poisson_probability(
                        returns_1,
                        self.in_lambda[0]
                    )

                    for returns_2 in range(self.MAX_ITERATION):
                        prob4 = prob3 * self.poisson_probability(
                            returns_2,
                            self.out_lambda[1]
                        )

                        cars = ncars

                        credits = 0
                        if -self.MAX_MOVE < movement < self.MAX_MOVE:
                            cars[0] -= movement
                            cars[1] += movement
                            if cars[0] < 0 or cars[1] < 0:
                                cars[0] += movement
                                cars[1] -= movement
                            else:
                                credits -= self.SHIFT_COST * movement

                        rented_1 = min(request_1, cars[0])
                        rented_2 = min(request_2, cars[1])

                        cars[0] = cars[0] + returns_1 - rented_1
                        cars[1] = cars[1] + returns_2 - rented_2

                        credits += self.RENT*(rented_1+rented_2)

                        if cars[0] > 20:
                            cars[0] = 20
                        elif cars[0] < 0:
                            credits = self.LOSE_PENALTY
                            cars[0] = 0
                        if cars[1] > 20:
                            cars[1] = 20
                        elif cars[1] < 0:
                            credits = self.LOSE_PENALTY
                            cars[1] = 0
                        result += prob4 * (credits + gamma * values[
                            state_to_index[tuple(cars)]
                        ])

        return (result)

    @staticmethod
    def poisson_probability(k_scalar, lambda_rate):

        lambda_pow_k = np.power(lambda_rate, k_scalar)

        exp_neg_lambda = np.exp(-lambda_rate)

        if k_scalar == 0:
            k_factorial = 1.0
        else:
            k_factorial = np.prod(np.arange(1, k_scalar + 1, dtype=np.double))

        probability = (lambda_pow_k * exp_neg_lambda) / k_factorial

        return probability


world = Env()
world.agent.show_values()
for _ in range(20):
    world.agent.evaluate_policy()
    world.agent.improve_policy()
    world.agent.show_values()
# print(Env.poisson_probability(10, 3))