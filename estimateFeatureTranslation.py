import numpy as np
from interp import interp2

# img1, img2: gray scale
def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2, windowSize):

    newX = startX
    newY = startY
    tol = 0.01
    midWindow = np.floor(windowSize/2)
    img1_window_start_x = int(startX-midWindow)
    img1_window_start_y = int(startY-midWindow)
    max_iter = 10

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

        A_pinv = np.linalg.pinv(A.transpose()@A)

        b = A.transpose()@b

        difference = A_pinv@b
        u = difference[0]
        v = difference[1]

        # updating
        ZX = newX + u
        ZY = newY + v
        if np.sqrt((ZX-newX)**2+(ZY-newY)**2)<tol:
            return ZX, ZY
        newX = ZX
        newY = ZY

    return newX, newY
