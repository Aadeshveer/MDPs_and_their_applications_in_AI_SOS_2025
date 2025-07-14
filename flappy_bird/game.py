import pygame as pg
import sys
from bird import Bird
from env import Pipes
from utils import load_image, load_images, Animation

WINDOW_SIZE = (300, 400)
SCREEN_SIZE = (600, 800)
SKY_BLUE = (207, 236, 247)


class Game:

    def __init__(self) -> None:
        self.bird: Bird = Bird()
        self.pipes: Pipes = Pipes(WINDOW_SIZE)

    def initialize_pygame(self):
        pg.init()
        self.screen = pg.display.set_mode(SCREEN_SIZE)
        self.window = pg.Surface(WINDOW_SIZE)
        self.clock = pg.time.Clock()
        self.assets = {
            'bird': Animation(load_images('./assets/images/bird')),
            'pipe': Animation([load_image('assets/images/pipe/pipe.png')]),
        }
        self.bird.initialize_interface(self.window, self.assets['bird'])
        self.pipes.initialize_interface(self.window, self.assets['pipe'])

    def play(self):
        self.initialize_pygame()
        ctr = 0
        while True:
            # print(self.bird.altitude)
            self.window.fill(SKY_BLUE)
            for event in pg.event.get():

                if event.type is pg.QUIT:
                    pg.quit()
                    sys.exit()

                if event.type == pg.KEYDOWN:

                    if event.key == pg.K_SPACE:
                        self.bird.flap()

            if ctr > 120:
                ctr = 0
                print('pipe spawned')
                print(self.pipes.pipe_list)
                self.pipes.spawn_pipe()
                print(self.pipes.pipe_list)
            ctr += 1
            self.bird.progress()
            self.bird.render()
            self.pipes.update()
            self.pipes.render()
            self.screen.blit(pg.transform.scale(self.window, SCREEN_SIZE))
            pg.display.update()
            self.clock.tick(60)


Game().play()
