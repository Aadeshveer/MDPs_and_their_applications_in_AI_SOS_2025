import numpy as np
import matplotlib.pyplot as plt
from k_armed_bandit import KArmedBandit, EpsilonGreedy

total_iterations = 10000
runs_per_iteration = 1000
bandit_arms = 10


def run_tests(
    machine_function,
    agent_function,
    total_itr=total_iterations,
    rpi=runs_per_iteration,
    UCB=False,
    gradient=False,
):
    # first array stores the reward and second stores optimal moves
    results_arr = np.zeros((2, rpi, total_itr))

    for i in range(total_itr):

        if i % 200 == 0:
            print(f'Running iteration {i}/{total_itr}')

        machine = machine_function()
        agent = agent_function()

        optimum_act = machine.get_optimal()

        for j in range(rpi):

            action = agent.choose_action(UCB, gradient)
            result = machine.draw(action)
            agent.update_table(action, reward=result, grad=gradient)

            results_arr[0, j, i] = result

            if action == optimum_act:
                results_arr[1, j, i] = 1

    return results_arr.mean(axis=2)


def machine_function_generator(stationary=True, bandit_arms=bandit_arms):
    return lambda: KArmedBandit(k=bandit_arms, stationary=stationary)


def agent_function_generator(
    eps=0.1,
    stationary=True,
    Q_initialization=0,
    alpha=0.1,
    baseline=0
):
    return lambda: EpsilonGreedy(
        epsilon=eps,
        stationary=stationary,
        Q_initialization=Q_initialization,
        alpha=alpha,
        baseline=baseline
    )


def compare_epsilon(epsilon_values=[0, 0.01, 0.1]):

    arr = np.zeros((len(epsilon_values), 2, runs_per_iteration))

    for i_eps, epsilon in enumerate(epsilon_values):

        print(f'Running tests for epsion {epsilon} ...')

        arr[i_eps] = run_tests(
            machine_function_generator(),
            agent_function_generator(epsilon, True)
        )

    plt.subplot(2, 1, 1)

    for i, epsilon in enumerate(epsilon_values):
        plt.plot(arr[i][0], label=f'Epsilon : {epsilon}')

    plt.title('Average performance of Epsilong greedy algorithm with various k')  # Noqa:E501
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()

    plt.subplot(2, 1, 2)

    for i, epsilon in enumerate(epsilon_values):
        plt.plot(arr[i][1], label=f'Epsilon : {epsilon}')

    plt.title('Fraction of optimal performance of Epsilong greedy algorithm with various k')  # Noqa:E501
    plt.xlabel('Steps')
    plt.ylabel('Optimal fraction')
    plt.legend()

    plt.tight_layout()
    plt.savefig('epsilon_greedy.png')


def compare_stationary():
    arr = np.zeros((3, 2, runs_per_iteration))

    print('Running tests for Stationary machine ...')

    arr[0] = run_tests(
        machine_function_generator(True),
        agent_function_generator(0.1, True)
    )

    print(
        'Running tests for non stationary machine and constant step size ...'
    )

    arr[1] = run_tests(
        machine_function_generator(False),
        agent_function_generator(0.1, False)
    )

    print('Running tests for non stationary machine and sample-averaging ...')

    arr[2] = run_tests(
        machine_function_generator(False),
        agent_function_generator(0.1, True)
    )

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)

    plt.plot(arr[0, 0], label='Stationary')
    plt.plot(arr[1, 0], label='Non Stationary with constant step size parameter')  # Noqa:E501
    plt.plot(arr[2, 0], label='Non Stationary with 1/n step size parameter')

    plt.title('Average performance of Epsilong greedy algorithm with non stationary input')  # Noqa:E501
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()

    plt.subplot(2, 1, 2)

    plt.plot(arr[0, 1], label='Stationary')
    plt.plot(arr[1, 1], label='Non Stationary with constant step size parameter')  # Noqa:E501
    plt.plot(arr[2, 1], label='Non Stationary with sample averaging')  # Noqa:E501

    plt.title('Fraction of optimal performance of Epsilong greedy algorithm with non stationary input')  # Noqa:E501
    plt.xlabel('Steps')
    plt.ylabel('Optimal fraction')
    plt.legend()

    plt.tight_layout()
    plt.savefig('epsilon_greedy_non_stationary.png')


def optimistic_greedy():
    arr = np.zeros((2, 2, runs_per_iteration))

    print('Running tests for Epsilon greedy agent ...')

    arr[0] = run_tests(
        machine_function_generator(),
        agent_function_generator(0.1)
    )

    print('Running tests for Optimistic greedy agent ...')

    arr[1] = run_tests(
        machine_function_generator(),
        agent_function_generator(0, Q_initialization=5)
    )

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)

    plt.plot(arr[0, 0], label='Non Optimistic')
    plt.plot(arr[1, 0], label='Optimistic greedy')

    plt.title('Average performance')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()

    plt.subplot(2, 1, 2)

    plt.plot(arr[0, 1], label='Non Optimistic')
    plt.plot(arr[1, 1], label='Optimistic greedy')

    plt.title('Fraction of optimal performance')
    plt.xlabel('Steps')
    plt.ylabel('Optimal fraction')
    plt.legend()

    plt.tight_layout()
    plt.savefig('optimistic_epsilon_greedy.png')


def upper_confidence_bound():

    arr = np.zeros((2, 2, runs_per_iteration))

    print('Running tests for Epsilon greedy agent ...')

    arr[0] = run_tests(
        machine_function_generator(),
        agent_function_generator(0.1),
    )

    print('Running tests for UCB agent ...')

    arr[1] = run_tests(
        machine_function_generator(),
        agent_function_generator(0),
        UCB=True,
    )

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)

    plt.plot(arr[0, 0], label='Epsilon greedy (0.1)')
    plt.plot(arr[1, 0], label='UCB (c=2)')

    plt.title('Average performance')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()

    plt.subplot(2, 1, 2)

    plt.plot(arr[0, 1], label='Epsilon greedy (0.1)')
    plt.plot(arr[1, 1], label='UCB (c=2)')

    plt.title('Fraction of optimal performance')
    plt.xlabel('Steps')
    plt.ylabel('Optimal fraction')
    plt.legend()

    plt.tight_layout()
    plt.savefig('UCB.png')


def gradient_bandit_algo():

    arr = np.zeros((5, 2, runs_per_iteration))

    print('Running tests for Epsilon greedy agent ...')

    arr[0] = run_tests(
        machine_function_generator(),
        agent_function_generator(0.1),
    )

    print('Running tests for gradient(0.1 baseless) agent ...')

    arr[1] = run_tests(
        machine_function_generator(),
        agent_function_generator(alpha=0.1),
        gradient=True,
    )

    print('Running tests for gradient(0.4 baseless) agent ...')

    arr[2] = run_tests(
        machine_function_generator(),
        agent_function_generator(alpha=0.4),
        gradient=True,
    )

    print('Running tests for gradient(0.1 base) agent ...')

    arr[3] = run_tests(
        machine_function_generator(),
        agent_function_generator(alpha=0.1, baseline=4),
        gradient=True,
    )

    print('Running tests for gradient(0.4 base) agent ...')

    arr[4] = run_tests(
        machine_function_generator(),
        agent_function_generator(alpha=0.4, baseline=4),
        gradient=True,
    )

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)

    plt.plot(arr[0, 0], label='Epsilon greedy (0.1)')
    plt.plot(arr[1, 0], label='alpha 0.1, baseless')
    plt.plot(arr[2, 0], label='alpha 0.4, baseless')
    plt.plot(arr[3, 0], label='alpha 0.1, with base')
    plt.plot(arr[4, 0], label='alpha 0.4, with base')

    plt.title('Average performance')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()

    plt.subplot(2, 1, 2)

    plt.plot(arr[0, 1], label='Epsilon greedy (0.1)')
    plt.plot(arr[1, 1], label='alpha 0.1, baseless')
    plt.plot(arr[2, 1], label='alpha 0.4, baseless')
    plt.plot(arr[3, 1], label='alpha 0.1, with base')
    plt.plot(arr[4, 1], label='alpha 0.4, with base')

    plt.title('Fraction of optimal performance')
    plt.xlabel('Steps')
    plt.ylabel('Optimal fraction')
    plt.legend()

    plt.tight_layout()
    plt.savefig('gradient.png')


compare_epsilon()
compare_stationary()
optimistic_greedy()
upper_confidence_bound()
gradient_bandit_algo()
