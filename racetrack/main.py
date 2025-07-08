import numpy as np
import matplotlib.pyplot as plt
import enum
import time
import random
from agent import Agent, MAX_SPEED, ACTION_MAP

BAR = '█'
FILLER = '▒'
TEST_SIZE = 100
TRAIN_ITERATION = 5000
NUM_ROUTES = 10


class MapElements(enum.Enum):
    WALL = 1
    OPEN = 0
    FINISH = 2
    START = 3


class Car:

    def __init__(self, num_pos: int) -> None:
        self.agent = Agent(num_pos)
        self.velocity = 0+0j
        self.pos = 0+0j

    def progress(self, idx: int):
        action = self.agent.choose_action(idx, self.velocity)
        self.velocity += action
        self.pos += self.velocity

    def process_episode(self):
        self.agent.update_values()


class Track:

    def __init__(self, map_idx: int = 1):
        self.map = np.loadtxt(f'maps/map{map_idx}.csv', delimiter=',')
        self.process_map()
        self.car = Car(len(self.idx_loc_map))
        self.reset_car_state()

    def reset_car_state(self):
        self.car.pos = random.choice(self.starting_points)
        self.car.velocity = 0+0j

    def process_map(self):
        self.idx_loc_map: dict[int, complex] = {}
        self.loc_idx_map: dict[complex, int] = {}
        self.starting_points: list[complex] = []
        self.finishing_points: list[complex] = []
        ctr = 0
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                match MapElements(self.map[i, j]):
                    case MapElements.WALL:
                        continue
                    case MapElements.START:
                        self.starting_points.append(complex(i, j))
                    case MapElements.FINISH:
                        self.finishing_points.append(complex(i, j))
                    case _:
                        pass
                self.loc_idx_map[complex(i, j)] = ctr
                self.idx_loc_map[ctr] = complex(i, j)
                ctr += 1

    def run_episode(self):

        self.reset_car_state()
        self.car.agent.start_new_episode()
        while self.car.pos not in self.finishing_points:
            old_car_position: complex = self.car.pos
            self.car.progress(self.loc_idx_map[old_car_position])
            self.validate_car_action(old_car_position)
        self.car.process_episode()

    def validate_transition(self, old_pos: complex) -> bool:
        new_pos = self.car.pos
        dr = new_pos-old_pos
        n_steps = int(max(abs(dr.real), abs(dr.imag)))
        if n_steps == 0:
            loc = tuple(map(int, (old_pos.real, old_pos.imag)))
            if self.map[*loc] == MapElements.WALL.value:
                return False
            return True
        increament = dr/n_steps
        for _ in range(n_steps+1):
            loc = tuple(map(round, (old_pos.real, old_pos.imag)))
            if self.map[*loc] == MapElements.WALL.value:
                return False
            old_pos += increament
        return True

    def validate_car_action(self, old_position: complex) -> bool:

        loc = tuple(map(int, (self.car.pos.real, self.car.pos.imag)))
        if not 0 <= loc[0] < self.map.shape[0]:
            self.reset_car_state()
            return False
        elif not 0 <= loc[1] < self.map.shape[1]:
            self.reset_car_state()
            return False
        elif not self.validate_transition(old_position):
            self.reset_car_state()
            return False

        vel_x = self.car.velocity.real
        vel_y = self.car.velocity.imag

        if vel_x < 0:
            vel_x = 0
        elif vel_x >= MAX_SPEED:
            vel_x = MAX_SPEED

        if vel_y < 0:
            vel_y = 0
        elif vel_y >= MAX_SPEED:
            vel_y = MAX_SPEED

        self.car.velocity = complex(vel_x, vel_y)

        if (self.car.velocity.real == 0 and
                self.car.velocity.imag == 0 and
                self.car.pos not in self.starting_points):
            self.reset_car_state()
            return False
        return True

    def train_model(self, iterations: int = TRAIN_ITERATION):
        print(f'running {iterations} iterations')
        print('\033[?25l', end="")
        print('[' + ' '*36, end='')
        for i in range(iterations):
            if i % 50 == 0:
                print('\b'*36, end='', flush=True)
                print((BAR*int(i/iterations * 20)).ljust(20, FILLER) + ']' + f' completed: {int(i/iterations*100):02}%', end='', flush=True)  # Noqa:E501
            self.run_episode()
        print('\b'*36, end='', flush=True)
        print(BAR*20+'] completed: 100%')
        print('\033[?25h', end="")
        self.car.agent.evaluate_policy()
        print(f'Total states                    : {self.car.agent.state_action_ctr.size}')  # Noqa:E501
        print(f'states arrived atleast once     : {(self.car.agent.state_action_ctr > 0).sum()}')  # Noqa:E501
        print(f'states arrived atleast twice    : {(self.car.agent.state_action_ctr > 1).sum()}')  # Noqa:E501
        print(f'states arrived atleast thrice   : {(self.car.agent.state_action_ctr > 2).sum()}')  # Noqa:E501
        print(f'states arrived more than 5 times: {(self.car.agent.state_action_ctr > 5).sum()}')  # Noqa:E501

    def test_model(self, num_tests: int = 100) -> int:
        print("Testing model")
        sum: int = 0
        for _ in range(num_tests):
            sum += self.test_once()
        return sum

    def test_once(self):
        self.reset_car_state()
        ctr = 0
        while self.car.pos not in self.finishing_points:
            ctr += 1
            action = self.car.agent.policy[
                self.loc_idx_map[self.car.pos],
                int(self.car.velocity.real),
                int(self.car.velocity.imag)
            ]
            old_car_position = self.car.pos
            self.car.velocity += ACTION_MAP[action]
            self.car.pos += self.car.velocity
            if not self.validate_car_action(old_car_position) or ctr > 100:
                return False
        return True

    def plot_motion(self, file_name: str | None, num_routes: int = NUM_ROUTES) -> None:
        positions: list[tuple[int, int]] = []
        plt.figure(figsize=(8, 8))
        plt.imshow(self.map)
        for j in range(num_routes):
            ctr = 0
            self.reset_car_state()
            while self.car.pos not in self.finishing_points and ctr < 200:
                ctr += 1
                action = self.car.agent.policy[
                    self.loc_idx_map[self.car.pos],
                    int(self.car.velocity.real),
                    int(self.car.velocity.imag)
                ]
                old_car_postion = self.car.pos
                self.car.velocity += ACTION_MAP[action]
                self.car.pos += self.car.velocity
                positions.append((
                    int(self.car.pos.real),
                    int(self.car.pos.imag)
                ))
                if not self.validate_car_action(old_car_postion):
                    break
            arr1 = np.zeros(len(positions))
            arr2 = np.zeros(len(positions))
            for i in range(arr1.size):
                arr1[i] = positions[i][0]
                arr2[i] = positions[i][1]
            plt.plot(arr2, arr1, marker='o', label=f'route{j+1}')
            positions = []
        plt.ylim(0, self.map.shape[0]-1)
        plt.xlim(0, self.map.shape[1]-1)
        plt.title('Routes followed by trained model')
        plt.legend()
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name)
        plt.close()


for i in range(0, 2):  # we have 2 maps
    print(f'\nSimulating track {i+1}\n')

    track = Track(i+1)

    image_ctr = 0

    score: int = track.test_model(TEST_SIZE)
    print(f'Model score with no training: {score}/{TEST_SIZE}\n')
    scores: list[int] = [score]
    averages: list[float] = [score]
    track.train_model()
    score: int = track.test_model(TEST_SIZE)
    print(f'Model score after training: {score}/{TEST_SIZE}\n')
    scores.append(score)
    averages.append(averages[-1] + (score-averages[-1])/len(scores))
    while score < TEST_SIZE*0.95:
        track.train_model()
        score = track.test_model(TEST_SIZE)
        print(f'Model score after training: {score}/{TEST_SIZE}\n')
        scores.append(score)
        averages.append(averages[-1] + (score-averages[-1])/len(scores))
        plt.plot(scores, label='scores')
        plt.plot(averages, label='average')
        plt.title('Performance of model')
        plt.xlabel(f'Num of training iterations completed(*{TRAIN_ITERATION})')
        plt.ylabel(f'Score of model(out of {TEST_SIZE})')
        plt.ylim(0, TEST_SIZE)
        plt.legend()
        plt.savefig(f'performace{i+1}.png')
        plt.close()
        track.plot_motion(f'results/map{i+1}.png')
    print(f'track {i+1} training complete')
    time.sleep(1)
