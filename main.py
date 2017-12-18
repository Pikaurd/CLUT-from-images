#!/usr/bin/env python3

from typing import Tuple
from functools import partial, lru_cache
from math import floor

import imageio
import scipy.misc
import numpy as np


width = 512
height = 512
channels = 3


def generate_identify_color_matrix():
    img = np.zeros((width, height, channels), dtype=np.uint8)
    for by in range(8):
        for bx in range(8):
            for g in range(64):
                for r in range(64):
                    x = r + bx * 64
                    y = g + by * 64
                    img[y][x][0] = int(r * 255.0 / 63.0 + 0.5)
                    img[y][x][1] = int(g * 255.0 / 63.0 + 0.5)
                    img[y][x][2] = int((bx + by * 8.0) * 255.0 / 63.0 + 0.5)
    return img


def generate_identify_color_matrix_v2():
    img = np.zeros((width, height, channels), dtype=np.uint8)



def generate_color_matrix_from_image(img, dest=None):
    lut = np.zeros((width, height, channels), dtype=np.uint8)
    img_list = img.tolist() if dest is None else dest.tolist()
    for iy in range(img.shape[1]):
        for ix in range(img.shape[0]):
            r, g, b = img_list[ix][iy]
            x, y, bx, by = color_coordinate(r, g, b)
            lut_y = y + by * 64
            lut_x = x + bx * 64
            lut[lut_y][lut_x][0] = r
            lut[lut_y][lut_x][1] = g
            lut[lut_y][lut_x][2] = b
            # print('{r} {g} {b} {x} {y} {bx} {by} {lut_x} {lut_y}'.format(r=r, g=g, b=b, x=x, y=y, bx=bx, by=by, lut_x=lut_x, lut_y=lut_y))
    return lut


@lru_cache(maxsize=512)
def color_coordinate(r, g, b) -> Tuple[int, int, int, int]:
    x, y, bx, by = 0, 0, 0, 0
    x = floor(r / 4.0)
    y = floor(g / 4.0)
    bx, by = blue_coordinate(floor(b / 4.0))
    return x, y, bx, by


@lru_cache(maxsize=64)
def blue_coordinate(b: int) -> Tuple[int, int]:
    assert b >= 0 and b <= 63, 'GOT {}'.format(b)
    x, y = 0, 0
    y = floor(floor(b) / 8.0)
    x = int(floor(b) - y * 8.0)
    return x, y


def show(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


def write(img, uri):
    imageio.imwrite(uri, img, 'PNG')


if __name__ == '__main__':
    import os
    import inspect

    current_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    dir_name = os.path.dirname(current_file_path)
    asset_dir = os.path.join(dir_name, 'asset')

    identity_lut = generate_identify_color_matrix()
    show(identity_lut)

    print(asset_dir)
