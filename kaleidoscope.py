#!/usr/bin/env python3

import cv2
import numpy as np


from functools import wraps
import time


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Time execution of module {} function {} is {} seconds".format(func.__module__, func.__name__, end - start))
        return result
    return wrapper


def core(img, N, out, r_start, r_out, c_in, c_out, scale, mode="full"):
    in_rows, in_cols = img.shape[:2]
    if c_in is None:
        c_in = (dim // 2 for dim in (in_rows, in_cols))
    c_y, c_x = c_in

    r_start %= 2 * np.pi
    width = np.pi / N
    r_end = r_start + width

    if out == 'same':
        out = np.empty((in_rows, in_cols, 3), dtype=np.uint8)
    elif out == 'full':
        quarter = np.pi / 2
        r_mid = (r_start + r_end) / 2
        if 0 <= r_mid < quarter:
            dy = in_rows - c_y
            dx = in_cols - c_x
        elif quarter <= r_mid <= 2 * quarter:
            dy = in_rows - c_y
            dx = c_x
        elif 2 * quarter <= r_mid <= 3 * quarter:
            dy = c_y
            dx = c_x
        else:
            dy = c_y
            dx = in_cols - c_x
        s = int(np.ceil(2 * np.sqrt(dx*dx + dy*dy) * scale))
        out = np.empty((s, s, 3), dtype=np.uint8)
    else:
        out = np.empty((out, out, 3), dtype=np.uint8)

    out_rows, out_cols = out.shape[:2]
    if c_out is None:
        c_out = (dim // 2 for dim in (out_rows, out_cols))
    co_y, co_x = c_out

    # create sample points and offset to center of output image
    Xp, Yp = np.meshgrid(np.arange(out_cols), np.arange(out_rows))
    Xp -= co_x
    Yp -= co_y

    # calculate magnitude and angle of each sample point in input image
    mag_p = np.sqrt(Xp*Xp + Yp*Yp) / scale
    theta_p = np.abs(((np.arctan2(Xp, Yp) - r_out) % (2 * width)) - width) \
              + r_start

    # convert to cartesian sample points in input image, offset by c_in
    Y = (mag_p * np.sin(theta_p) + c_y).astype(np.int64)
    X = (mag_p * np.cos(theta_p) + c_x).astype(np.int64)

    # set outside valid region pixels to black (avoid index error)
    # temporarily use pixel [0,0] of input image
    old = img[0,0].copy()
    img[0,0] = (0, 0, 0)

    if mode == "circle":
        bad = (Y < 0) | (Y >= in_rows) | (X < 0) | (X >= in_cols)
        Y[bad] = 0
        X[bad] = 0
    elif mode == "full":
        Y = np.where(Y < 0, - (Y % in_rows), Y)
        Y = np.where(Y > in_rows, (in_rows - (Y % in_rows)), Y)
        X = np.where(X < 0, - (X % in_cols), X)
        X = np.where(X > in_cols, (in_cols - (X % in_cols)), X)
        Y = np.where(Y == in_rows, (in_rows - 1), Y)
        X = np.where(X == in_cols, (in_cols - 1), X)
    else:
        raise ValueError("Unknown mode")

    # sample input image to set each pixel of out
    out[:] = img[Y, X]

    img[0,0] = old # restore input [0,0] to its initial value
    return out, c_x, c_y, r_start, r_end


def add_annotation(img, c_x, c_y, r_start, r_end):
    in_rows, in_cols = img.shape[:2]
    # draw a circle at the input c_in
    cv2.circle(img, (c_x, c_y), 10, (0,0,255), 2)
    # draw lines from c_in to display sample region in input image
    l = min(max(c_x, in_cols-c_x), max(c_y, in_rows-c_y)) / 3
    cv2.line(img, (c_x, c_y), (int(c_x + l*np.cos(r_start)),
                               int(c_y + l*np.sin(r_start))),
             (255,0,0), 2)
    cv2.line(img, (c_x, c_y), (int(c_x + l * np.cos(r_end)),
                               int(c_y + l * np.sin(r_end))),
             (0,255,0), 2)


#@timethis
def kaleido(img, N=10, out='same', r_start=0, r_out=0, c_in=None, c_out=None,
            scale=1, annotate=False, mode="full"):
    ''' Return a kaleidoscope from img, with specified parameters.
    'img' is a 3-channel uint8 numpy array of image pixels.
    'N' is the number of mirrors.
    'out' can be 'same', 'full', or a 3-channel uint8 array to fill.
    'r_start' is the selection rotation from the input image [clock radians].
    'r_out' is the rotation of the output image result [clock radians].
    'c_in' is the origin point of the sample sector from the input image.
        If None defaults to the center of the input image [c_y,c_x].
    'c_out' is the center of the kaleidoscope in the output image. If None
        defaults to the center point of the output image [c_y, c_x].
    'scale' is the scale of the output kaleidoscope. Default 1.
    'annotate' is a boolean denoting whether to annotate the input image to
        display the selected region. Default True.
    '''
    out, c_x, c_y, r_start, r_end = core(img, N, out, r_start, r_out,
                                         c_in, c_out, scale, mode)

    if annotate:
        add_annotation(img, c_x, c_y, r_start, r_end)

    return out


#if __name__ == '__main__':
#    from argparse import ArgumentParser
#
#    parser = ArgumentParser()
#    parser.add_argument('-f', '--filename', default='test.png',
#                        help='path to image file')
#    parser.add_argument('-n', type=int, default=15, help='number of mirrors')
#    parser.add_argument('-o', '--out', default='full',
#                        choices=('same', 'full'),
#                        help='output size '
#                             '(same as input or full kaleidoscope)')
#    parser.add_argument('--r_start', type=float, default=0.6,
#                        help='clockwise radians rotation of input image')
#    parser.add_argument('--r_out', type=float, default=0,
#                        help='clockwise radians rotation of output image')
#    parser.add_argument('--c_in', nargs=2, type=int,
#                        help='c_y c_x - origin point of the sample sector from'
#                             ' the input image')
#    parser.add_argument('--c_out', nargs=2, type=int,
#                        help='c_y c_x - center point of the kaleidoscope in '
#                             'the output image')
#    parser.add_argument('-s', '--scale', type=float, default=1,
#                        help='scale of the output kaleidoscope')
#    parser.add_argument('-a', '--annotate', action='store_true')
#
#    args = parser.parse_args()
#
#    image = cv2.imread(args.filename)
#    out = kaleido(image, args.n, args.out, args.r_start, args.r_out, args.c_in,
#                  args.c_out, args.scale, args.annotate)
#
#    cv2.imshow('in', image)
#    cv2.imshow('out', out)
#    cv2.waitKey()
#
#    cv2.destroyAllWindows()
#