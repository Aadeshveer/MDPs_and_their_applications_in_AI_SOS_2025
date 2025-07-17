import torch
import random
# import numpy as np
from collections import deque
from model import NeuralNet, QLearning

MAX_MEMORY = 100_000
MIN_EPSILON = 0.001
EPSILON_DECAY = 0.995
BATCH_SIZE = 1000
LR = 0.0001
EPSILON = 1
GAMMA = 0.99
HIDDEN_SIZE = 128
INPUT_SIZE = 4  # to accomodate State
OUTPUT_SIZE = 2  # idx 0 for no flap and 1 for flap


class State:

    def __init__(
        self,
        alt_diff: float,
        bird_vel: float,
        pipe_dist: float,
        bird_alt: float
    ) -> None:
        self.alt_diff = alt_diff
        self.bird_vel = bird_vel
        self.pipe_dist = pipe_dist
        self.bird_alt = bird_alt

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor((
            self.alt_diff,
            self.bird_vel,
            self.pipe_dist,
            self.bird_alt
        ), dtype=torch.float32)


class MemoryPoint:

    def __init__(
        self,
        state: State,
        action: bool,
        reward: float,
        new_state: State,
        done: bool
    ) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.new_state = new_state
        self.done = done

    def to_column_tensor(self) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:
        return (
            self.state.to_tensor().reshape((1, -1)),
            torch.tensor(self.action, dtype=torch.float32).reshape((1, -1)),
            torch.tensor(self.reward, dtype=torch.float32).reshape((1, -1)),
            self.new_state.to_tensor().reshape((1, -1)),
            torch.tensor(self.done, dtype=torch.float32).reshape((1, -1)),
        )


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON
        self.memory: deque[MemoryPoint] = deque(maxlen=MAX_MEMORY)
        self.model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.trainer = QLearning(self.model, LR, GAMMA)

    def choose_action(self, state: State):
        if random.random() > self.epsilon:
            values: torch.Tensor = self.model(state.to_tensor())
            return values.argmax().item() == 1
        else:
            return random.random() < 0.5

    def add_to_memory(self, memory: MemoryPoint):
        self.memory.append(memory)

    def train_one_data(self, memory: MemoryPoint):
        self.trainer.train(
            *memory.to_column_tensor()
        )

    def train_multiple_data(self):
        self.epsilon = max(self.epsilon*EPSILON_DECAY, MIN_EPSILON)
        if len(self.memory) < 2:
            return
        if len(self.memory) > BATCH_SIZE:
            training_set = random.sample(self.memory, BATCH_SIZE)
        else:
            training_set = list(self.memory)

        data: list[torch.Tensor] = []
        for tensors in zip(*map(lambda x: x.to_column_tensor(), training_set)):
            data.append(torch.cat(tensors, dim=0))
        self.trainer.train(*data)
