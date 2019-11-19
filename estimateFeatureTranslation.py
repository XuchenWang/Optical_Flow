"""
[newX, newY] = estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2)
• (INPUT) startX: Represents the starting X coordinate for a single feature in the first frame
• (INPUT) startY: Represents the starting Y coordinate for a single feature in the first frame
• (INPUT) Ix: H × W matrix representing the gradient along the X-direction
• (INPUT) Iy: H × W matrix representing the gradient along the Y-direction
• (INPUT)img1:H×W×3matrixrepresentingthefirstimageframe
• (INPUT)img2:H×W×3matrixrepresentingthesecondimageframe
• (OUTPUT) newX: Represents the new X coordinate for a single feature in the second frame
• (OUTPUT) newY: Represents the new Y coordinate for a single feature in the second frame
"""

import numpy as np
from interp import interp2

# img1, img2: gray scale
def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2, windowSize):
    newX = startX
    newY = startY
    tol = 0.001
    error = 1
    midWindow = np.floor(windowSize/2)
    img1_window_start_x = int(startX-midWindow)
    img1_window_start_y = int(startY-midWindow)

    while (error > tol):
        window_start_x = int(newX-midWindow)
        window_start_y = int(newY-midWindow)
        window_end_x = int(window_start_x+windowSize)
        window_end_y = int(window_start_y+windowSize)

        x_range = np.arange(window_start_x,window_end_x)
        y_range = np.arange(window_start_y,window_end_y)

        xx,yy = np.meshgrid(x_range,y_range)

        Ix_window = interp2(Ix,yy,xx)
        Iy_window = interp2(Iy,yy,xx)

        It_window = img1[img1_window_start_y:(img1_window_start_y+windowSize), \
                    img1_window_start_x:(img1_window_start_x+windowSize)] - interp2(img2,yy,xx)

        #Build up Ax=-b and solve it to find (u,v)
        second_moment=np.zeros([2,2])
        second_moment[0,0] = np.sum(Ix_window*Ix_window)
        second_moment[0,1] = np.sum(Ix_window*Iy_window)
        second_moment[1,0] = np.sum(Iy_window*Ix_window)
        second_moment[1,1] = np.sum(Iy_window*Iy_window)
        b = np.zeros([2,1])
        b[0,0] = np.sum(Ix_window*It_window)
        b[1,0] = np.sum(Iy_window*It_window)
        b = -b
        inverse_second_moment = np.linalg.inv(second_moment)
        difference = np.dot(inverse_second_moment, b)
        u = difference[0,0]
        v = difference[1,0]

        # updating
        newX = newX + u
        newY = newY + v
        error = np.linalg.norm(It_window)

    return newX, newY
