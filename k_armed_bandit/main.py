import numpy as np
import matplotlib.pyplot as plt
from k_armed_bandit import KArmedBandit, EpsilonGreedy

epsilon_values = [0, 0.01, 0.1]
total_iterations = 1000
runs_per_iteration = 2000
bandit_arms = 10

results_arr = np.zeros(
    (
        len(epsilon_values),
        runs_per_iteration,
        total_iterations,
    )
)

optimal_arr = np.zeros(
    (
        len(epsilon_values),
        runs_per_iteration,
        total_iterations,
    )
)

for i_eps, epsilon in enumerate(epsilon_values):

    print(f'Running tests for epsion {epsilon} ...')

    machine = KArmedBandit(k=bandit_arms)

    for i in range(total_iterations):

        if i % 200 == 0:
            print(f'Running iteration {i}/{total_iterations}')

        agent = EpsilonGreedy(epsilon=epsilon, k=bandit_arms)

        optimum_act = machine.get_optimal()

        for j in range(runs_per_iteration):

            action = agent.choose_action()
            result = machine.draw(action)
            agent.update_table(action, reward=result)

            results_arr[i_eps, j, i] = result

            if action == optimum_act:
                optimal_arr[i_eps, j, i] = 1

avg_result = results_arr.mean(axis=2)
optimal_result = optimal_arr.mean(axis=2)

plt.subplot(2, 1, 1)

for i, epsilon in enumerate(epsilon_values):
    plt.plot(avg_result[i], label=f'Epsilon : {epsilon}')

plt.title('Average performance of Epsilong greedy algorithm with various k')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()

plt.subplot(2, 1, 2)

for i, epsilon in enumerate(epsilon_values):
    plt.plot(optimal_result[i], label=f'Epsilon : {epsilon}')

plt.title('Fraction of optimal performance of Epsilong greedy algorithm with various k')  # Noqa:E501
plt.xlabel('Steps')
plt.ylabel('Optimal fraction')
plt.legend()

plt.tight_layout()
plt.savefig('epsilon_greedy.png')
