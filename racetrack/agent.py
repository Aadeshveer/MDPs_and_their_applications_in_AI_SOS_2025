import numpy as np
import random

ACTION_MAP = {
    0: 1+0j,
    1: 1+1j,
    2: 0+1j,
    3: -1+1j,
    4: -1+0j,
    5: -1+-1j,
    6: 0+-1j,
    7: 1+-1j,
    8: 0+0j,
}

MAX_SPEED = 5
EPSILON = 0.1
GAMMA = 0.9


class Agent:

    def __init__(self, num_pos: int) -> None:
        self.npos = num_pos
        # there are 9 actions possible
        self.state_action_values = np.random.random((
            num_pos,
            1+MAX_SPEED,
            1+MAX_SPEED,
            9
        )) - 100
        self.state_action_ctr = np.zeros((
            self.npos,
            1+MAX_SPEED,
            1+MAX_SPEED,
            9
        ))
        self.policy = np.random.randint(low=0, high=9, size=(
            num_pos, 1 + MAX_SPEED, 1 + MAX_SPEED
        ), dtype=int)

        self.SARlist: list[tuple[tuple[int, complex], int, int]] = []

    def start_new_episode(self):
        self.SARlist = []

    def evaluate_policy(self):
        self.policy = self.state_action_values.argmax(axis=3)

    def choose_action(
        self,
        state_idx: int,
        velocity: complex
    ) -> complex:
        if random.random() > EPSILON:
            action = self.policy[
                state_idx,
                int(velocity.real),
                int(velocity.imag)
            ]
        else:
            action = random.randint(0, 8)
        self.SARlist.append(((state_idx, velocity), action, -1))
        return ACTION_MAP[action]

    def update_values(self):
        self.SARlist.reverse()
        g = 0
        w = 1
        for state, action, reward in self.SARlist:
            state_idx = (
                state[0],
                int(state[1].real),
                int(state[1].imag)
            )
            g = GAMMA * g + reward
            self.state_action_ctr[*state_idx, action] += w

            update = g - self.state_action_values[*state_idx, action]
            update *= w/self.state_action_ctr[*state_idx, action]

            self.state_action_values[*state_idx, action] += update

            new_val = self.state_action_values[*state_idx].argmax()  # Noqa:E501
            # if (self.policy[*state_idx] != new_val):
            #     print(f'policy updated for state{state_idx} to {new_val}')
            self.policy[*state_idx] = new_val

            if self.policy[*state_idx] != action:
                break

            w *= 9/(9-8*EPSILON)
