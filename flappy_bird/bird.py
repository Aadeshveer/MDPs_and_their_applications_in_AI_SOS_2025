import pygame as pg
from utils import Animation

BIRD_SIZE = (16, 12)
MAX_UP_VEL = 3
GRAVITY = 0.15
SCALE = 1


class Bird:

    def __init__(self, window_size: tuple[int, int]) -> None:
        self.altitude: float = 0
        self.fall_vel: float = 0
        self.window_size = window_size

    def flap(self):
        self.fall_vel = -MAX_UP_VEL
        self.assets.update(2)

    def progress(self):
        self.altitude += self.fall_vel * SCALE
        self.fall_vel += GRAVITY

    def update_anim(self):
        if self.fall_vel > MAX_UP_VEL/2:
            self.assets.update(0)
        elif self.fall_vel > 0:
            self.assets.update(1)

    def initialize_interface(
        self,
        window: pg.Surface,
        assets: Animation,
    ):
        self.window = window
        self.assets = assets

    def render(self):
        self.assets.render(
            self.window,
            (
                self.window.get_width()//4,
                int(self.altitude),
            )
        )

    def out_of_window(self) -> bool:
        if self.altitude < 0:
            return True
        if self.altitude > self.window_size[1] - BIRD_SIZE[1]:
            return True
        return False
