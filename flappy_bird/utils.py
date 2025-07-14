import pygame as pg
import os


def load_image(path_to_image: str) -> pg.Surface:
    image_object = pg.image.load(path_to_image).convert()
    image_object.set_colorkey((0, 0, 0))
    return image_object


def load_images(images_folder: str) -> list[pg.Surface]:
    images: list[pg.Surface] = []
    for obj_path in os.listdir(images_folder):
        images.append(load_image(os.path.join(images_folder, obj_path)))
    return images


class Animation:

    def __init__(self, assets: list[pg.Surface]) -> None:
        self.surfaces = assets
        self.frame_ctr = 0
        self.frame_interval = 10
        self.pointer: int = 0

    def render(self, display: pg.Surface, location: tuple[int, int]):
        self.frame_ctr += 1
        display.blit(self.surfaces[self.pointer], location)

    def update(self, frame: int = -1):
        if frame < 0:
            if self.frame_ctr > self.frame_interval:
                self.pointer += 1
                self.frame_ctr = 0
                self.pointer %= len(self.surfaces)
        else:
            self.pointer = frame
