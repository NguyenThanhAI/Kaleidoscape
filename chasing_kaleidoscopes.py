import os
import argparse

import random
import time

import imageio

from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

import numpy as np

from scipy import ndimage

import cv2

from kaleidoscope import kaleido, core

from PIL import Image


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--n_kaleidos", type=int, default=100)
    parser.add_argument("--min_num_sides", type=int, default=5)
    parser.add_argument("--max_num_sides", type=int, default=35)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--target_size", type=int, default=320)
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--n_proc", type=int, default=6)
    parser.add_argument("--mode", type=str, choices=("circle", "full"), default="full")
    parser.add_argument("--save_dir", type=str, default=".")

    args = parser.parse_args()

    return args


def gen_params(n_kaleidos, height, width, min_num_sides, max_num_sides):
    r_start = np.linspace(0, 2 * np.pi, num=n_kaleidos)
    c_y = np.linspace(int(0.1 * height), int(0.9 * height), num=n_kaleidos, dtype=np.int).tolist()
    c_x = np.linspace(int(0.1 * width), int(0.9 * width), num=n_kaleidos, dtype=np.int).tolist()
    c_in = list(zip(c_y, c_x))
    N = np.random.randint(min_num_sides, max_num_sides + 1)

    return r_start, c_in, N


def gen_kaleidoscope(i, image, r_start, c_in, N, target_size, scale, mode):
    time_image = image.copy()
    out = kaleido(img=time_image, N=N, out="full", r_start=r_start[i], r_out=0, c_in=c_in[i],
                  c_out=None, scale=scale, annotate=False, mode=mode)
    out = cv2.resize(out, (target_size, target_size))

    return Image.fromarray(out[:, :, ::-1])


if __name__ == '__main__':
    args = get_args()

    image_path = args.image_path
    n_kaleidos = args.n_kaleidos
    min_num_sides = args.min_num_sides
    max_num_sides = args.max_num_sides
    fps = args.fps
    target_size = args.target_size
    scale = args.scale
    n_proc = args.n_proc
    mode = args.mode
    save_dir = args.save_dir

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    #r_start = np.linspace(0, 2 * np.pi, num=n_kaleidos)
    #c_y = np.linspace(int(0.1 * height), int(0.9 * height), num=n_kaleidos, dtype=np.int).tolist()
    #c_x = np.linspace(int(0.1 * width), int(0.9 * width), num=n_kaleidos, dtype=np.int).tolist()
    #c_in = list(zip(c_y, c_x))
    #N = np.random.randint(5, 25)

    r_start, c_in, N = gen_params(n_kaleidos=n_kaleidos, height=height, width=width,
                                  min_num_sides=min_num_sides, max_num_sides=max_num_sides)

    gen_kaleidoscope_step = partial(gen_kaleidoscope, image=image, r_start=r_start,
                                    c_in=c_in, N=N, target_size=target_size, scale=scale, mode=mode)

    with Pool(processes=n_proc) as pool:
        images_list = pool.map(gen_kaleidoscope_step, list(range(n_kaleidos)))

    duration = np.ceil(n_kaleidos / fps)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    images_list[0].save(os.path.join(save_dir, os.path.splitext(os.path.basename(image_path))[0] + ".gif"),
                        save_all=True, append_images=images_list[1:], loop=0, duration=duration)