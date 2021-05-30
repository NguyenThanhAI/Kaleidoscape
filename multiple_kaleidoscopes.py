import os

import random
import time

from tqdm import tqdm

import numpy as np

from scipy import ndimage

import cv2

from kaleidoscope import kaleido, core


image_path = "deijvh7-3599e286-d8dd-4cb2-8b21-2726cc76a9ed.jpg"
image = cv2.imread(image_path)

n_kaleidos = 20

height, width = image.shape[:2]

#for i in range(n_kaleidos):
#    time_image = image.copy()
#    N = np.random.randint(5, 52)
#    r_start = np.random.randn() * np.pi * 0.5
#    r_out = np.random.randn() * np.pi * 0.5
#    c_in = (np.random.randint(int(0.4 * height), int(0.6 * height)), np.random.randint(int(0.4 * width), int(0.6 * width)))
#    out = kaleido(img=time_image, N=N, out="full", r_start=r_start, r_out=r_out, c_in=c_in,
#                  c_out=None, scale=0.2, annotate=False)
#
#    y, x = np.nonzero(np.sum(out, axis=-1))
#    y_min = np.min(y)
#    y_max = np.max(y)
#    x_min = np.min(x)
#    x_max = np.max(x)
#    print(y_min, y_max, x_min, x_max)
#
#    out = out[y_min:(y_max + 1), x_min:(x_max + 1)].copy()
#    print(N, r_start, r_out, c_in, out.shape)
#    cv2.imshow("Result", out)
#    cv2.imshow("Original image", time_image)
#
#    cv2.waitKey(2000)
#
#    cv2.destroyAllWindows()

N = np.random.randint(5, 52)
r_start = np.random.randn() * np.pi * 0.5
r_out = np.random.randn() * 2 * np.pi * 0.5
c_in = (np.random.randint(int(0.2 * height), int(0.8 * height)), np.random.randint(int(0.2 * width), int(0.8 * width)))

time_image = image.copy()
out = kaleido(img=time_image, N=N, out="full", r_start=r_start, r_out=r_out, c_in=c_in,
              c_out=None, scale=0.2, annotate=False)

y, x = np.nonzero(np.sum(out, axis=-1))
y_min = np.min(y)
y_max = np.max(y)
x_min = np.min(x)
x_max = np.max(x)
print(y_min, y_max, x_min, x_max)

out = out[y_min:(y_max + 1), x_min:(x_max + 1)].copy()
#out = cv2.resize(out, (250, 250), interpolation=cv2.INTER_CUBIC)

cv2.imshow("Result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

angles = np.linspace(0, -360, num=100)
#print(angles)


def rotateImage(image, angle):
    center=tuple(np.array(image.shape[0:2])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[0:2],flags=cv2.INTER_LINEAR)

for angle in tqdm(angles):
    start = time.time()
    rotated = rotateImage(out, angle)
    #print(rotated.shape)
    cv2.imshow("Result", rotated)
    cv2.waitKey(25)
