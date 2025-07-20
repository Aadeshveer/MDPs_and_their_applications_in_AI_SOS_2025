import pygame as pg
import sys
from bird import Bird, BIRD_SIZE, MAX_UP_VEL
from env import Pipes, Pipe, CLEARANCE
from utils import load_image, load_images, Animation
from agent import Agent, State, MemoryPoint
from plotter import Plotter

WINDOW_SIZE = (300, 400)
SCREEN_SIZE = (600, 800)
SKY_BLUE = (207, 236, 247)
FPS = 60
STEP_CONTROL = 4


class Game:

    def __init__(self) -> None:
        self.bird: Bird = Bird(WINDOW_SIZE)
        self.pipes: Pipes = Pipes(WINDOW_SIZE)

    def initialize_AI(self):
        self.agent = Agent()
        self.plotter = Plotter()

    def reset(self):
        self.bird = Bird(WINDOW_SIZE)
        self.pipes = Pipes(WINDOW_SIZE)

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
                break
            if self.bird.out_of_window():
                break
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
            self.clock.tick(FPS)
        print('\n')
        print(' GAME OVER '.center(40, '*'))
        print('\033[?25h', end="")

    def get_state(self) -> State:
        closest_next_pipe: Pipe = self.pipes.get_next_pipe()
        return State(
            (self.bird.altitude - closest_next_pipe.height + CLEARANCE)/WINDOW_SIZE[1],  # Noqa:E501
            (self.bird.altitude + BIRD_SIZE[1] - closest_next_pipe.height)/WINDOW_SIZE[1],  # Noqa:E501
            (self.bird.altitude)/WINDOW_SIZE[1],
            (WINDOW_SIZE[1] - self.bird.altitude + BIRD_SIZE[1])/WINDOW_SIZE[1],  # Noqa:E501
            self.bird.fall_vel/MAX_UP_VEL,
            closest_next_pipe.x/WINDOW_SIZE[0],
        )

    def AI_game(self, interface: bool = False, log_file: str | None = None):
        score = 0
        n = 0
        fps = 10000
        if log_file is not None:
            with open(log_file, 'w') as f:
                # print(alert) # IMPORTANT changed w to r
                f.write('score,episode_length\n')
        if interface:
            self.initialize_pygame()
        print('getting ai')
        self.initialize_AI()
        print('begining')
        while True:
            n += 1
            # if n % 100 == 0:
            #     self.plotter.plot('performance.png')
            done = False
            score = 0
            self.reset()
            if interface:
                self.bird.initialize_interface(
                    self.window, self.assets['bird']
                )
                self.pipes.initialize_interface(
                    self.window, self.assets['pipe']
                )
            ctr = 0
            while not done:
                if ctr % 120 == 0:
                    self.pipes.spawn_pipe()

                if interface:
                    self.window.fill(SKY_BLUE)
                    for event in pg.event.get():
                        if event.type is pg.QUIT:
                            pg.quit()
                            sys.exit()
                        if event.type == pg.KEYDOWN:
                            if event.key == pg.K_SPACE:
                                if fps == 60:
                                    fps = 10000
                                else:
                                    fps = 60

                ctr += 1
                done = False
                state = self.get_state()
                if ctr % STEP_CONTROL == 0:
                    action = self.agent.choose_action(state)
                else:
                    action = False
                result = self.iterate(action)
                match result:
                    case -1:
                        reward = -50
                        done = True
                    case 1:
                        score += result
                        reward = 1000
                    case 0:
                        reward = 0.1
                    case -2:
                        reward = -100
                        done = True
                    case _:
                        raise ValueError(f'result given by iterate method is faulty: {result}')  # Noqa:E501
                memory = MemoryPoint(
                    state,
                    action,
                    reward,
                    self.get_state(),
                    done
                )
                # self.agent.train_one_data(memory)
                self.agent.add_to_memory(memory)
                if interface:
                    self.bird.render()
                    self.pipes.render()
                    self.bird.update_anim()
                    self.screen.blit(
                        pg.transform.scale(self.window, SCREEN_SIZE)
                    )
                    pg.display.update()
                    self.clock.tick(fps)
                if done:
                    break

            if n % 10 == 0:
                self.agent.update_target_model()
                print(f'{n}. score: {score}, iterations: {ctr}, epsilon: {self.agent.epsilon}')

            self.agent.train_multiple_data()
            if log_file is not None:
                with open(log_file, 'a') as f:
                    f.write(f'{score},{ctr}\n')
            if n % 1000 == 0:
                self.agent.model.save('model.pth')
            # self.plotter.add_score(score, ctr)

    def iterate(self, action: bool) -> int:
        if action:
            self.bird.flap()
        self.bird.progress()
        if self.pipes.check_collision(int(self.bird.altitude)):
            return -1
        if self.bird.out_of_window():
            return -2
        score_update = self.pipes.update()
        return score_update


if __name__ == '__main__':
    game = Game()
    game.AI_game(log_file='log1.csv')
