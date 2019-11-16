import cv2
import createMask
import numpy as np
import matplotlib.pyplot as plt
import imutils
from PIL import Image
def boundingBox(firstFrame, maskNum):
    coor_matrix = np.zeros((4,maskNum))

    for ii in range(maskNum):
        mask = createMask.main(firstFrame)
        i,j = np.where(mask==1)
        # compute the x_min,x_max,y_min,y_max
        y_min = np.min(i)
        y_max = np.max(i)

        x_min = np.min(j)
        x_max = np.max(j)
        # construct matrix
        w = x_max - x_min
        h = y_max - y_min
        coor_matrix[0,ii] = x_min
        coor_matrix[1,ii] = y_min
        coor_matrix[2,ii] = w
        coor_matrix[3,ii] = h
        print(coor_matrix)
    return coor_matrix
if __name__ == '__main__':
    coor_matrix = boundingBox('firstFrame.jpg', 2)
    print(coor_matrix)
