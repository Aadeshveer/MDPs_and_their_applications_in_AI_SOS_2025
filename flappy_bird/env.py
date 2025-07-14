import pygame as pg
from utils import Animation
import random

GAME_VEL = 1
PIPE_LEN = 512
CLEARANCE = 64


class Pipes:

    def __init__(self, window_size: tuple[int, int]) -> None:
        self.window_size = window_size
        self.pipe_list: list[Pipe] = []
        self.interface = False

    def initialize_interface(self, window: pg.Surface, asset: Animation):
        self.interface = True
        self.assets = asset
        self.window = window
        self.interface = True
        for pipe in self.pipe_list:
            pipe.initialize_interface(window, asset)

    def spawn_pipe(self):
        new_pipe = Pipe(self.window_size)
        if self.interface:
            new_pipe.initialize_interface(self.window, self.assets)
        self.pipe_list.append(new_pipe)

    def update(self):
        for pipe in self.pipe_list:
            if not pipe.update():
                self.pipe_list.remove(pipe)

    def render(self):
        for pipe in self.pipe_list:
            pipe.render()


class Pipe:

    def __init__(self, window_size: tuple[int, int]) -> None:
        self.x: int = window_size[1]
        self.height: int = random.randint(
            window_size[0]//4,
            3*window_size[0]//4 + CLEARANCE
        )
        self.window_size = window_size

    def initialize_interface(
        self,
        window: pg.Surface,
        assets: Animation,
    ):
        self.window = window
        self.assets = assets

    def update(self):
        self.x -= GAME_VEL
        if self.x < -self.window_size[0]:
            return False
        return True

    def render(self):
        self.assets.render(self.window, (self.x, self.height))
        self.assets.render(self.window, (self.x, self.height - PIPE_LEN - CLEARANCE))
