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
    # # pad the images and coordinator
    # img1 = np.pad(img1, ((windowSize, windowSize), (windowSize, windowSize)), 'constant')
    # img2 = np.pad(img2, ((windowSize, windowSize), (windowSize, windowSize)), 'constant')
    # Ix = np.pad(Ix, ((windowSize, windowSize), (windowSize, windowSize)), 'constant')
    # Iy = np.pad(Iy, ((windowSize, windowSize), (windowSize, windowSize)), 'constant')
    # startX = startX + windowSize
    # startY = startY + windowSize

    newX = startX
    newY = startY
    tol = 0.1
    # error = 1
    midWindow = np.floor(windowSize/2)
    img1_window_start_x = int(startX-midWindow)
    img1_window_start_y = int(startY-midWindow)
    max_iter = 100
    for _ in range(max_iter):
        window_start_x = int(newX-midWindow)
        window_start_y = int(newY-midWindow)
        window_end_x = int(window_start_x+windowSize)
        window_end_y = int(window_start_y+windowSize)

        x_range = np.arange(window_start_x,window_end_x)
        y_range = np.arange(window_start_y,window_end_y)

        xx,yy = np.meshgrid(x_range,y_range)

        Ix_window = interp2(Ix,xx,yy)
        Iy_window = interp2(Iy,xx,yy)

        It_window = interp2(img2,xx,yy) - img1[img1_window_start_y:(img1_window_start_y+windowSize), \
                    img1_window_start_x:(img1_window_start_x+windowSize)]

        #Build up Ax=-b and solve it to find (u,v)
        A = np.zeros([100,2])
        A[:,0] = Ix_window.flatten()
        A[:,1] = Iy_window.flatten()
        b = It_window.flatten()
        b = -b

        # difference = np.linalg.solve(A.transpose()@A,A.transpose()@b)

        A_pinv = np.linalg.pinv(A.transpose()@A)

        b = A.transpose()@b

        difference = A_pinv@b





        # second_moment=np.zeros([2,2])
        # second_moment[0,0] = np.sum(Ix_window*Ix_window)
        # second_moment[0,1] = np.sum(Ix_window*Iy_window)
        # second_moment[1,0] = np.sum(Iy_window*Ix_window)
        # second_moment[1,1] = np.sum(Iy_window*Iy_window)
        # b = np.zeros([2,1])
        # b[0,0] = np.sum(Ix_window*It_window)
        # b[1,0] = np.sum(Iy_window*It_window)
        # b = -b
        # inverse_second_moment = np.linalg.inv(second_moment)
        # difference = np.dot(inverse_second_moment, b)

        # difference = np.linalg.lstsq(second_moment,b)[0]
        u = difference[0]
        v = difference[1]

        # updating
        # newX = newX + u
        # newY = newY + v

        ZX = newX + u
        ZY = newY + v

        if np.sqrt((ZX-newX)**2+(ZY-newY)**2)<tol:
            return ZX, ZY

        newX = ZX
        newY = ZY

        # error = np.linalg.norm(It_window)

    # newX = newX-windowSize
    # newY = newY-windowSize
    return newX, newY
