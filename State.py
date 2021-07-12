import numpy as np
import sys
import cv2


class State():
    def __init__(self, size, move_range):
        self.image = np.zeros(size, dtype=np.float32)
        self.move_range = move_range

    def reset(self, x):
        self.image = x

    def step(self, act):
        neutral = 6
        move = act.astype(np.float32)
        moves = (move - neutral) / 20
        moved_image = np.zeros(self.image.shape, dtype=np.float32)
        # de = move[:, 3:, :, :]
        r = self.image[:, 0, :, :]
        g = self.image[:, 1, :, :]
        b = self.image[:, 2, :, :]
        moved_image[:, 0, :, :] = r + (moves[:, 0, :, :]) * r * (1 - r)
        moved_image[:, 1, :, :] = g + (0.1 * moves[:, 1, :, :] + 0.9 * moves[:, 0, :, :]) * g * (1 - g)
        moved_image[:, 2, :, :] = b + (0.1 * moves[:, 2, :, :] + 0.9 * moves[:, 0, :, :]) * b * (1 - b)

        # gaussian = np.zeros(self.image.shape, self.image.dtype)
        # bilateral = np.zeros(self.image.shape, self.image.dtype)
        # median = np.zeros(self.image.shape, self.image.dtype)
        # b, c, h, w = self.image.shape
        # # for i in range(0, b):
        # #     if np.sum(de[i] == 0) > 0:
        # #         gaussian[i] = cv2.GaussianBlur(moved_image[i], ksize=(5, 5), sigmaX=0.5)
        # #     if np.sum(de[i] == 1) > 0:
        # #         bilateral[i] = cv2.bilateralFilter(moved_image[i], d=5, sigmaColor=0.1, sigmaSpace=5)
        # #     if np.sum(de[i] == 2) > 0:
        # #         median[i] = cv2.medianBlur(moved_image[i], ksize=5)
        #
        # self.image = np.where(de == 0, gaussian, moved_image)
        # self.image = np.where(de == 1, bilateral, moved_image)
        # self.image = np.where(de == 2, median, moved_image)
        '''
        gaussian = np.zeros(self.image.shape, self.image.dtype)
        bilateral = np.zeros(self.image.shape, self.image.dtype)
        median = np.zeros(self.image.shape, self.image.dtype)
        b, c, h, w = self.image.shape
        gaussian = np.zeros(self.image.shape, self.image.dtype)
        gaussian2 = np.zeros(self.image.shape, self.image.dtype)
        bilateral = np.zeros(self.image.shape, self.image.dtype)
        bilateral2 = np.zeros(self.image.shape, self.image.dtype)
        median = np.zeros(self.image.shape, self.image.dtype)
        box = np.zeros(self.image.shape, self.image.dtype)
        b, c, h, w = self.image.shape
        for i in range(0,b):
            if np.sum(act[i]==self.move_range) > 0:
                gaussian[i,0] = cv2.GaussianBlur(self.image[i,0], ksize=(5,5), sigmaX=0.5)
            if np.sum(act[i]==self.move_range+1) > 0:
                bilateral[i,0] = cv2.bilateralFilter(self.image[i,0], d=5, sigmaColor=0.1, sigmaSpace=5)
            if np.sum(act[i]==self.move_range+2) > 0:
                median[i,0] = cv2.medianBlur(self.image[i,0], ksize=5)
            if np.sum(act[i]==self.move_range+3) > 0:
                gaussian2[i,0] = cv2.GaussianBlur(self.image[i,0], ksize=(5,5), sigmaX=1.5)
            if np.sum(act[i]==self.move_range+4) > 0:
                bilateral2[i,0] = cv2.bilateralFilter(self.image[i,0], d=5, sigmaColor=1.0, sigmaSpace=5)
            if np.sum(act[i]==self.move_range+5) > 0:
                box[i,0] = cv2.boxFilter(self.image[i,0], ddepth=-1, ksize=(5,5))
        '''
        self.image = 0.8 * moved_image + 0.2 * self.image

        '''
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range, gaussian, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+1, bilateral, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+2, median, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+3, gaussian2, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+4, bilateral2, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+5, box, self.image)
        '''
