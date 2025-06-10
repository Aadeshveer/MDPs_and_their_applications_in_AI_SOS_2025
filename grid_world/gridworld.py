import numpy as np
import pygame
import enum
import random
import sys
from MDP import GridWorldMDP, Action

SIDE = 16
FPS = 60


class TileTypes(enum.Enum):
    NORMAL = 0
    LAVA = -10
    TREASURE = 10
    COIN = 1
    WALL = -1


class GridWorld:

    def __init__(self, mapping=1, stochastic_prob=None, gamma=0.9):

        if mapping is None:
            self.mapping = np.loadtxt('maps/map1.txt', delimiter=',')
        else:
            self.mapping = np.loadtxt(f'maps/map{mapping}.txt', delimiter=',')

        self.SIZE = self.mapping.shape
        self.WINDOWSIZE = (
            self.SIZE[0] * SIDE,
            self.SIZE[1] * SIDE,
        )

        self.agent = GridWorldMDP(
            self.mapping,
            TileTypes.NORMAL.value,
            self.act,
            gamma
        )

        if stochastic_prob is None:
            self.stochastic_prob = 1
        else:
            self.stochastic_prob = stochastic_prob

        self.reset()

        self.reward = 0
        self.old_pos = (1, 1)

    def initialize_UI(self):

        pygame.init()

        pygame.display.set_caption('Grid World MDP')

        self.load_assets()

        self.window = pygame.display.set_mode(self.WINDOWSIZE)

        self.clock = pygame.Clock()

        self.print_grid()

    def reset(self):
        self.truck_pos = (1, 1)
        self.truck_facing = Action.RIGHT

        self.need_change = False

    def run(self):

        self.initialize_UI()

        while (True):

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    match event.key:
                        case pygame.K_LEFT:
                            self.act(Action.LEFT)
                        case pygame.K_UP:
                            self.act(Action.UP)
                        case pygame.K_RIGHT:
                            self.act(Action.RIGHT)
                        case pygame.K_DOWN:
                            self.act(Action.DOWN)

            if self.need_change:
                self.print_change()

            pygame.display.update()
            self.clock.tick(FPS)

    def act(self, action, from_pos=None):
        self.truck_facing = action
        self.old_pos = self.truck_pos
        action_list = [
            Action.LEFT,
            Action.UP,
            Action.RIGHT,
            Action.DOWN
        ]
        if from_pos is None:
            pos = list(self.truck_pos)
        else:
            pos = list(from_pos)
        if self.stochastic_prob < random.random():
            action_list.remove(action)
            action = random.choice(action_list)
        match action:
            case Action.LEFT:
                pos[0] -= 1
            case Action.UP:
                pos[1] -= 1
            case Action.RIGHT:
                pos[0] += 1
            case Action.DOWN:
                pos[1] += 1

        reward = self.mapping[*pos]

        if from_pos is None:
            self.reward = reward
            match self.reward:
                case TileTypes.WALL.value:
                    self.truck_pos = self.old_pos
                case TileTypes.LAVA.value:
                    self.reset()
                case TileTypes.TREASURE.value:
                    self.reset()
                case TileTypes.COIN.value:
                    self.reset()
                case _:
                    self.truck_pos = tuple(pos)

        else:
            match reward:
                case TileTypes.WALL.value:
                    pos = self.old_pos
                case TileTypes.LAVA.value:
                    pos = (1, 1)
                case TileTypes.TREASURE.value:
                    pos = (1, 1)
                case TileTypes.COIN.value:
                    pos = (1, 1)
                case _:
                    pos = tuple(pos)

        self.need_change = True
        return (pos, reward)

    def print_change(self):
        if self.need_change:
            self.print_tile(*self.old_pos)
            self.window.blit(
                self.truck_images[self.truck_facing.value],
                self.pos_loc(self.truck_pos)
            )
            self.need_change = False

    def print_grid(self):

        for i in range(self.SIZE[0]):
            for j in range(self.SIZE[1]):
                self.print_tile(i, j)

    def print_tile(self, i, j):
        if ((i, j) == self.truck_pos):
            self.window.blit(
                self.truck_images[self.truck_facing.value],
                (i*SIDE, j*SIDE)
            )
        elif (self.mapping[i][j] == TileTypes.NORMAL.value):
            self.window.blit(random.choice(self.normal_tile), (i*SIDE, j*SIDE))
        elif (self.mapping[i][j] == TileTypes.LAVA.value):
            self.window.blit(random.choice(self.lava_tile), (i*SIDE, j*SIDE))
        elif (self.mapping[i][j] == TileTypes.TREASURE.value):
            self.window.blit(self.treasure_tile, (i*SIDE, j*SIDE))
        elif (self.mapping[i][j] == TileTypes.COIN.value):
            self.window.blit(self.coin_tile, (i*SIDE, j*SIDE))
        elif (self.mapping[i][j] == TileTypes.WALL.value):
            self.window.blit(random.choice(self.wall_tile), (i*SIDE, j*SIDE))

    def pos_loc(self, pos):
        return (pos[0] * SIDE, pos[1] * SIDE)

    def load_assets(self):

        self.truck_images = []
        for i in range(4):
            self.truck_images.append(
                pygame.image.load(f'images/truck{i+1}.png')
            )

        self.normal_tile = []
        self.normal_tile.append(
            pygame.image.load('images/tiles1.png')
        )
        self.normal_tile.append(
            pygame.image.load('images/tiles2.png')
        )
        self.normal_tile.append(
            pygame.image.load('images/tiles3.png')
        )
        for i in range(3):
            self.normal_tile.append(
                pygame.transform.rotate(self.normal_tile[0], 90+i*90)
            )
            self.normal_tile.append(
                pygame.transform.rotate(self.normal_tile[1], 90+i*90)
            )
            self.normal_tile.append(
                pygame.transform.rotate(self.normal_tile[2], 90+i*90)
            )

        self.lava_tile = []
        self.lava_tile.append(
            pygame.image.load('images/tiles6.png')
        )
        self.lava_tile.append(
            pygame.image.load('images/tiles7.png')
        )
        for i in range(3):
            self.lava_tile.append(
                pygame.transform.rotate(self.lava_tile[0], 90+i*90)
            )
            self.lava_tile.append(
                pygame.transform.rotate(self.lava_tile[1], 90+i*90)
            )

        self.wall_tile = []
        self.wall_tile.append(
            pygame.image.load('images/tiles4.png')
        )
        self.wall_tile.append(
            pygame.image.load('images/tiles5.png')
        )
        for i in range(3):
            self.wall_tile.append(
                pygame.transform.rotate(self.wall_tile[0], 90+i*90)
            )
            self.wall_tile.append(
                pygame.transform.rotate(self.wall_tile[1], 90+i*90)
            )

        self.treasure_tile = pygame.image.load('images/tiles8.png')

        self.coin_tile = pygame.image.load('images/tiles9.png')


def PI_VI_on_maps():
    for i in range(5):
        print(f"Processing map: {i+1}, Default setting.")
        world = GridWorld(i+1)
        world.agent.iterate_values()
        world.agent.show_values(f'results/map{i+1}/value_iteration.png')
        world = GridWorld(i+1)
        world.agent.iterate_policy()
        world.agent.show_policy(f'results/map{i+1}/policy_iteration.png')


def test():
    for i in range(5):
        print(f"Processing map: {i+1}, test setting.")
        world = GridWorld(i+1, stochastic_prob=0.8, gamma=0.1)
        world.agent.iterate_values()
        world.agent.improve_policy()
        world.agent.show_policy(
            f'results/map{i+1}/highslip_lowgamma_policy.png',
            0.2
        )
        world.agent.show_values(
            f'results/map{i+1}/highslip_lowgamma_values.png',
            0.2
        )


PI_VI_on_maps()
test()
