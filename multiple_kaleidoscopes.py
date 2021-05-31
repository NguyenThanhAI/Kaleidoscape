import os
import argparse

import random
import time

import imageio

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


def rotateImage(image, angle):
    center=tuple(np.array(image.shape[0:2])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[0:2],flags=cv2.INTER_LINEAR)


def init_kaleidoscopes(image, n_kaleidos, min_num_sides, max_num_sides, show_image, scale):
    height, width = image.shape[:2]

    kaleidos_list = []

    for i in range(n_kaleidos):
        time_image = image.copy()
        N = np.random.randint(min_num_sides, max_num_sides)
        r_start = np.random.randn() * np.pi * 0.5
        r_out = np.random.randn() * np.pi * 0.5
        c_in = (np.random.randint(int(0.4 * height), int(0.6 * height)), np.random.randint(int(0.4 * width), int(0.6 * width)))
        out = kaleido(img=time_image, N=N, out="full", r_start=r_start, r_out=r_out, c_in=c_in,
                      c_out=None, scale=0.2, annotate=False)

        y, x = np.nonzero(np.sum(out, axis=-1))
        y_min = np.min(y)
        y_max = np.max(y)
        x_min = np.min(x)
        x_max = np.max(x)
        #print(y_min, y_max, x_min, x_max)

        out = out[y_min:(y_max + 1), x_min:(x_max + 1)].copy()
        kaleidos_list.append(out)
        #print(N, r_start, r_out, c_in, out.shape)
        #cv2.imshow("Result", out)
        #cv2.imshow("Original image", time_image)
        #cv2.waitKey(2000)
        #cv2.destroyAllWindows()

    max_shape = max(kaleidos_list, key=lambda x: np.prod(x.shape)).shape
    min_shape = min(kaleidos_list, key=lambda x: np.prod(x.shape)).shape
    kaleidos_list.sort(key=lambda x: np.prod(x.shape), reverse=True)
    kaleidos_shape = list(map(lambda x: x.shape[:2], kaleidos_list))
    if len(kaleidos_list) > 1:
        result_image = np.zeros(shape=(int(max_shape[0] * 2), int(max_shape[1] * 2), 3), dtype=np.uint8)
    else:
        result_image = out.copy()
    accepted_mask = np.ones(shape=(result_image.shape[0], result_image.shape[1]), dtype=np.bool)

    result_height, result_width = result_image.shape[:2]
    #print(result_height, result_width)
    chosen_coords = []
    if len(kaleidos_list) > 1:
        for kaleidoscope in kaleidos_list:
            kalei_height, kalei_width = kaleidoscope.shape[:2]
            accepted_y = result_height - kalei_height
            accepted_x = result_width - kalei_width
            #while True:
            #    chosen_y = np.random.randint(accepted_y)
            #    chosen_x = np.random.randint(accepted_x)
            #    if accepted_mask[chosen_y, chosen_x] is True:
            #        accepted_mask[chosen_y: (chosen_y + kalei_height), chosen_x: (chosen_x + kalei_width)] = False
            #        result_image[chosen_y: (chosen_y + kalei_height), chosen_x: (chosen_x + kalei_width)] = kaleidoscope
            #        break
            #    else:
            #        continue
            chosen_y = np.random.randint(accepted_y)
            chosen_x = np.random.randint(accepted_x)
            result_image[chosen_y: (chosen_y + kalei_height), chosen_x: (chosen_x + kalei_width)] = np.where(kaleidoscope, kaleidoscope, result_image[chosen_y: (chosen_y + kalei_height), chosen_x: (chosen_x + kalei_width)])
            #print(chosen_y, chosen_x)
            chosen_coords.append((chosen_y, chosen_x))
    else:
        chosen_coords.append((0, 0))

    result_image = cv2.resize(result_image, (int(result_image.shape[1] / scale), int(result_image.shape[0] / scale)))
    if show_image:
        cv2.imshow("Result", result_image)
        cv2.waitKey(0)

    return kaleidos_list, chosen_coords, max_shape


def init_angles_list(n_kaleidos, min_num_angle_steps, max_num_angle_steps):
    angles_list = []
    for j in range(n_kaleidos):
        if np.random.rand() < 0.5:
            angles_list.append(np.linspace(0, 360, num=np.random.randint(min_num_angle_steps, max_num_angle_steps + 1)))
        else:
            angles_list.append(np.linspace(0, -360, num=np.random.randint(min_num_angle_steps, max_num_angle_steps + 1)))

    min_angle_steps = len(min(angles_list, key=lambda x: len(x)))
    #print(min_angle_steps)

    return angles_list, min_angle_steps


def generate_images_list(kaleidos_list, chosen_coords, max_shape, angles_list, min_angle_steps, scale):
    result_images_list = []

    for k in range(min_angle_steps):
        #start = time.time()
        if len(kaleidos_list) > 1:
            result_image = np.zeros(shape=(int(max_shape[0] * 2), int(max_shape[1] * 2), 3), dtype=np.uint8)
        else:
            result_image = np.zeros(shape=(int(max_shape[0]), int(max_shape[1]), 3), dtype=np.uint8)
        for kaleidoscope, chosen_coord, angles in zip(kaleidos_list, chosen_coords, angles_list):
            kalei_height, kalei_width = kaleidoscope.shape[:2]
            if k >= len(angles):
                continue
            rotated = rotateImage(kaleidoscope, angles[k]).transpose(1, 0, 2)
            #print(rotated.shape, kaleidoscope.shape)
            chosen_y, chosen_x = chosen_coord
            result_image[chosen_y: (chosen_y + kalei_height), chosen_x: (chosen_x + kalei_width)] = np.where(rotated, rotated, result_image[chosen_y: (chosen_y + kalei_height), chosen_x: (chosen_x + kalei_width)])
        #end = time.time()
        #duration = int((end - start) * 1000.)
        result_image = cv2.resize(result_image, (int(result_image.shape[1] / scale), int(result_image.shape[0] / scale)))
        result_images_list.append(Image.fromarray(result_image[:, :, ::-1]))
        if show_image:
            cv2.imshow("Anh", result_image)
            #if duration > time_per_step:
            #    wait_time = 1
            #else:
            #    wait_time = duration - time_per_step
            #cv2.waitKey(wait_time)
            cv2.waitKey(1)

    return result_images_list


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--min_num_kaleidos", type=int, default=3)
    parser.add_argument("--max_num_kaleidos", type=int, default=6)
    parser.add_argument("--min_num_sides", type=int, default=5)
    parser.add_argument("--max_num_sides", type=int, default=25)
    parser.add_argument("--min_num_angle_steps", type=int, default=30)
    parser.add_argument("--max_num_angle_steps", type=int, default=100)
    parser.add_argument("--fps", type=int, default=80)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--show_image", type=str2bool, default=True)
    parser.add_argument("--save_dir", type=str, default=".")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    image_path = args.image_path
    n_kaleidos = np.random.randint(args.min_num_kaleidos, args.max_num_kaleidos + 1)

    min_num_sides = args.min_num_sides
    max_num_sides = args.max_num_sides

    max_num_angle_steps = args.max_num_angle_steps
    min_num_angle_steps = args.min_num_angle_steps

    fps = args.fps
    scale = args.scale

    show_image = args.show_image
    save_dir = args.save_dir
    time_per_step = int((1. / fps) * 1000.)

    image = cv2.imread(image_path)

    kaleidos_list, chosen_coords, max_shape = init_kaleidoscopes(image=image, n_kaleidos=n_kaleidos,
                                                                 min_num_sides=min_num_sides,
                                                                 max_num_sides=max_num_sides, show_image=show_image,
                                                                 scale=scale)

    angles_list, min_angle_steps = init_angles_list(n_kaleidos=n_kaleidos,
                                                    min_num_angle_steps=min_num_angle_steps,
                                                    max_num_angle_steps=max_num_angle_steps)

    result_images_list = generate_images_list(kaleidos_list=kaleidos_list, chosen_coords=chosen_coords,
                                              max_shape=max_shape, angles_list=angles_list,
                                              min_angle_steps=min_angle_steps, scale=scale)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    result_images_list[0].save(os.path.join(save_dir, os.path.splitext(os.path.basename(image_path))[0] + ".gif"), save_all=True, append_images=result_images_list[1:], loop=0)
