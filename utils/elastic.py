from PIL import Image
import numpy as np
import cv2
import random


class ElasticTransform:

    def __init__(self, alpha=40, sigma=6, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img):

        if random.random() > self.p:
            return img

        # PIL → numpy
        img = np.array(img)

        shape = img.shape[:2]

        dx = cv2.GaussianBlur(
            (np.random.rand(*shape) * 2 - 1),
            (17, 17),
            self.sigma
        ) * self.alpha

        dy = cv2.GaussianBlur(
            (np.random.rand(*shape) * 2 - 1),
            (17, 17),
            self.sigma
        ) * self.alpha

        x, y = np.meshgrid(
            np.arange(shape[1]),
            np.arange(shape[0])
        )

        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        distorted = cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        # ✅ numpy → PIL（关键！！）
        return Image.fromarray(distorted)