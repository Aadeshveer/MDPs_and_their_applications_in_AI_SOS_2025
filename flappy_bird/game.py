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
        self.bird: Bird = Bird(WINDOW_SIZE)
        self.pipes: Pipes = Pipes(WINDOW_SIZE)

    def initialize_pygame(self):
        pg.init()
        pg.display.set_caption('Flappy bird')
        self.screen = pg.display.set_mode(SCREEN_SIZE)
        self.window = pg.Surface(WINDOW_SIZE)
        self.clock = pg.time.Clock()
        self.assets = {
            'bird': Animation(load_images('./assets/images/bird')),
            'pipe': Animation([load_image('./assets/images/pipe/pipe.png')]),
            'title': Animation(
                [load_image('./assets/images/title/title.png')]
            ),
            'title_anim': Animation(
                load_images('./assets/images/title/title_anim')
            ),
        }
        self.bird.initialize_interface(self.window, self.assets['bird'])
        self.pipes.initialize_interface(self.window, self.assets['pipe'])

    def play(self):
        self.initialize_pygame()
        ctr = 0
        score = 0
        print('\n')
        print(' GAME START '.center(40, '*'))
        print('\n'*1)
        print('\033[?25l', end="")
        print(' '*15 + 'score : 00', end='', flush=True)
        while True:
            self.window.fill(SKY_BLUE)
            for event in pg.event.get():

                if event.type is pg.QUIT:
                    pg.quit()
                    sys.exit()

                if event.type == pg.KEYDOWN:

                    if event.key == pg.K_SPACE:
                        self.bird.flap()

            if ctr % 120 == 0:
                self.pipes.spawn_pipe()
            ctr += 1
            self.bird.progress()
            self.bird.render()
            if self.pipes.check_collision(int(self.bird.altitude)):
                return
            if self.bird.out_of_window():
                return
            score += self.pipes.update()
            self.pipes.render()
            self.bird.update_anim()
            if ctr < 120:
                self.assets['title'].render(self.window, (0, 0))
                self.assets['title_anim'].render(self.window, (0, 0))
                self.bird.altitude = self.window.get_height() // 2
                self.bird.fall_vel = 0
                self.assets['title_anim'].update()
            self.screen.blit(pg.transform.scale(self.window, SCREEN_SIZE))
            pg.display.update()
            print('\b\b', end='', flush=True)
            print(f'{score}'.rjust(2, '0'), end='', flush=True)
            self.clock.tick(60)


g = Game()
g.play()
print('\n')
print(' GAME OVER '.center(40, '*'))
print('\033[?25h', end="")
