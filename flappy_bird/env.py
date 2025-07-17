import pygame as pg
from utils import Animation
import random

GAME_VEL = 1
PIPE_LEN = 512
PIPE_WIDTH = 16
CLEARANCE = 64
BIRD_SIZE = (16, 12)


class Pipe:

    def __init__(self, window_size: tuple[int, int]) -> None:
        self.x: int = window_size[1] // 4 + 120
        self.worth = 1
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

    def update(self) -> bool:
        self.x -= GAME_VEL
        if self.x < -self.window_size[0]:
            return False
        return True

    def render(self):
        self.assets.render(self.window, (self.x, self.height))
        self.assets.render(
            self.window,
            (self.x, self.height - PIPE_LEN - CLEARANCE)
        )


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

    def update(self) -> int:
        score: int = 0
        for pipe in self.pipe_list:
            if not pipe.update():
                self.pipe_list.remove(pipe)
            if not self.window_size[0] // 4 - PIPE_WIDTH < pipe.x:
                score += pipe.worth
                pipe.worth = 0
        return score

    def render(self):
        for pipe in self.pipe_list:
            pipe.render()

    def check_collision(self, bird_altitude: int):
        for pipe in self.pipe_list:
            right_clearance = self.window_size[0] // 4 - PIPE_WIDTH < pipe.x
            left_clearance = pipe.x < self.window_size[0] // 4 + BIRD_SIZE[0]
            if left_clearance and right_clearance:
                if pipe.height < bird_altitude + BIRD_SIZE[1]:
                    return True
                elif pipe.height - CLEARANCE > bird_altitude:
                    return True
        return False

    def get_next_pipe(self) -> Pipe:
        for pipe in self.pipe_list:
            if pipe.worth > 0:
                return pipe
        return self.pipe_list[0]
