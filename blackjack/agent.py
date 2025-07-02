from plot import draw_policy_and_value, plot_value_function
from cards import Card, Hand
import numpy as np
import random


class Agent:

    def __init__(self, control: bool = False) -> None:
        self.control = control
        # 10 values for dealer card, 10(12-21) for self sum, 1 for ace
        # note for sum in range 4 to 11 player should always hit
        # Choose to hit for value 1 in policy
        if self.control:
            self.state_action_values = np.zeros((10, 10, 2, 2))
            self.state_action_ctr = np.zeros((10, 10, 2, 2), dtype=int)
            self.policy = np.ones((10, 10, 2, 2)) / 2
        else:
            self.values = np.zeros((10, 10, 2))
            self.state_ctr = np.zeros((10, 10, 2), dtype=int)
            self.policy = np.zeros((10, 10, 2, 2), dtype=np.int8)
            self.policy[:, 8:, :, 0] = 1
            self.policy[:, :8, :, 1] = 1
        # SAR = [(state(dealer, sum, ace), action, reward),...]
        self.SAR_list: list[tuple[tuple[int, int, bool], int, int]] = []
        self.GAMMA = 1

    def choose_action(
        self,
        dealer_card: Card,
        hand: Hand,
        rand: bool = False
    ) -> int:

        sum = hand.sum()
        dealer_show = min(dealer_card.value, 10)
        usable_ace = hand.usable_ace()

        if rand:
            action = random.randint(0, 1)
        else:
            if sum <= 11 or self.policy[
                dealer_show-1,
                sum-12,
                int(usable_ace),
                1
            ] > random.random():
                action = 1
            else:
                action = 0
        self.SAR_list.append(((dealer_show-1, sum-12, usable_ace), action, 0))
        return action

    def evaluate_value_function(self, final_reward: int) -> None:
        self.SAR_list.reverse()
        final_state, final_action, _ = self.SAR_list[0]
        self.SAR_list[0] = (final_state, final_action, final_reward)

        g = 0
        for state, action, reward in self.SAR_list:
            g = self.GAMMA * g + reward
            state_loc = (
                state[0],
                state[1],
                int(state[2]),
            )

            if state_loc[1] >= 0:
                if self.control:
                    self.state_action_ctr[*state_loc, action] += 1
                    Q_k = self.state_action_values[*state_loc, action]
                    update = (g - Q_k)/self.state_action_ctr[*state_loc, action]  # Noqa:E501
                    self.state_action_values[*state_loc, action] += update
                    q0 = self.state_action_values[*state_loc, 0]
                    q1 = self.state_action_values[*state_loc, 1]
                    if q1 > q0:
                        target = np.array((0, 1))
                    elif q1 < q0:
                        target = np.array((1, 0))
                    else:
                        target = np.array((0.5, 0.5))
                    self.policy[*state_loc] += (target - self.policy[*state_loc]) * 0.1  # Noqa:E501
                else:
                    self.state_ctr[*state_loc] += 1
                    V_k = self.values[*state_loc]
                    self.values[*state_loc] += (g - V_k)/self.state_ctr[*state_loc]  # Noqa:E501

    def draw_value_function(self, title: str):
        plot_value_function(title, self.values)

    def draw_policy(self, title: str):
        draw_policy_and_value(title, self.policy, self.state_action_values)

    def reset(self) -> None:
        self.SAR_list = []
