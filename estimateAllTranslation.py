"""
[newXs, newYs] = estimateAllTranslation(startXs,startYs,img1,img2)
• (INPUT) startXs: N × F matrix representing the starting X coordinates of all the features in the first frame for
all the bounding boxes
• (INPUT) startYs: N × F matrix representing the starting Y coordinates of all the features in the first frame for all the bounding boxes
• (INPUT)img1:H×W×3matrixrepresentingthefirstimageframe
• (INPUT)img2:H×W×3matrixrepresentingthesecondimageframe
• (OUTPUT) newXs: N × F matrix representing the new X coordinates of all the features in all the bounding boxes
• (OUTPUT) newYs: N × F matrix representing the new Y coordinates of all the features in all the bounding boxes
"""
import numpy as np
import scipy.signal as signal
import helper
from estimateFeatureTranslation import estimateFeatureTranslation

# startXs: N × F
# img1, img2: H×W
def estimateAllTranslation(startXs, startYs, img1, img2):
    newXs = startXs.copy()
    newYs = startYs.copy()
    N,F = startXs.shape

    # compute derivative
    dx, dy = np.gradient(helper.GaussianPDF_2D(0, 1, 10, 10), axis = (1,0))
    Ix = signal.convolve2d(img1,dx,'same')
    Iy = signal.convolve2d(img1,dy,'same')
    for f in range(F):
        for n in range(N):
            startX = startXs[n,f]
            startY = startYs[n,f]
            new_x, new_y = estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2, windowSize=10)
            newXs[n,f] = new_x
            newYs[n,f] = new_y

    return newXs, newYs
